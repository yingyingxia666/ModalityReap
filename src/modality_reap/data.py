from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torchaudio
from transformers import AutoProcessor

logger = logging.getLogger(__name__)

DEFAULT_DATASET_PREFIX = "/data/szs/share/voice_model_project/datasets"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    modality: str
    path: str
    format: str = "auto"


DEFAULT_DATASET_SPECS: list[DatasetSpec] = [
    DatasetSpec(
        "AudioCaps",
        "audio",
        "/data/szs/share/voice_model_project/datasets/AudioCaps/AudioCaps_train_AudioCaptioning.jsonl",
    ),
    DatasetSpec(
        "Clotho",
        "audio",
        "/data/szs/share/voice_model_project/datasets/Clotho/Clotho_train_AudioCaptioning.jsonl",
    ),
    DatasetSpec(
        "AISHELL-3",
        "audio",
        "/data/szs/share/voice_model_project/datasets/AISHELL-3/aishell3_train_tts.jsonl",
    ),
    DatasetSpec(
        "VCTK",
        "audio",
        "/data/szs/share/voice_model_project/datasets/VCTK/vctk_tts.jsonl",
    ),
    DatasetSpec(
        "LibriTTS",
        "audio",
        "/data/szs/share/voice_model_project/datasets/LibriTTS/libritts_train_clean_100_tts.jsonl",
    ),
    DatasetSpec(
        "UltraChat",
        "text",
        "/data/szs/share/voice_model_project/datasets/ultrachat/ultrachat_train_text_sft.jsonl",
    ),
    DatasetSpec(
        "OpenHermes-2.5",
        "text",
        "/data/szs/share/voice_model_project/datasets/OpenHermes-2___5/openhermes2_5_processed.jsonl",
    ),
    DatasetSpec(
        "tulu-3-sft-mixture",
        "text",
        "/data/szs/share/voice_model_project/datasets/tulu-3-sft-mixture/jsonl/train-00000-of-00006.jsonl",
    ),
]


AUDIO_CORE_EXPERTS = {8, 16, 26, 69, 78, 81}
AUDIO_SECONDARY_EXPERTS = {49, 52, 56, 57, 65, 75, 82, 103, 105, 122}
CROSS_MODAL_EXPERTS = {8, 52, 65, 75, 105}


class DataAvailabilityError(FileNotFoundError):
    pass



def build_dataset_specs(dataset_root: str | None = None) -> list[DatasetSpec]:
    if not dataset_root or dataset_root == DEFAULT_DATASET_PREFIX:
        return DEFAULT_DATASET_SPECS

    remapped: list[DatasetSpec] = []
    for spec in DEFAULT_DATASET_SPECS:
        try:
            relative = Path(spec.path).relative_to(DEFAULT_DATASET_PREFIX)
        except ValueError:
            relative = Path(spec.path).name
        remapped.append(
            DatasetSpec(spec.name, spec.modality, str(Path(dataset_root) / relative), spec.format)
        )
    return remapped



def _resolve_audio_path(audio_path: str, base_path: str) -> str:
    if not audio_path:
        return audio_path
    if os.path.isabs(audio_path):
        return audio_path
    return os.path.abspath(os.path.join(base_path, audio_path))



def _normalize_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""

    parts: list[str] = []
    for item in content:
        if isinstance(item, dict):
            if item.get("audio_path"):
                parts.append("<audio>")
            text = item.get("text")
            if text:
                parts.append(str(text).strip())
        else:
            value = str(item).strip()
            if value:
                parts.append(value)
    return "\n".join(part for part in parts if part).strip()



def normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for message in messages:
        normalized.append(
            {
                "role": message.get("role", "user"),
                "content": _normalize_message_content(message.get("content")),
            }
        )
    return normalized



def _extract_audio_paths_from_messages(messages: list[dict[str, Any]], base_path: str) -> list[str]:
    paths: list[str] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            audio_path = item.get("audio_path")
            if audio_path:
                paths.append(_resolve_audio_path(audio_path, base_path))
    return paths



def load_audio_tensor(audio_path: str, target_sample_rate: int) -> torch.Tensor | None:
    if not audio_path or not os.path.exists(audio_path):
        return None
    try:
        audio_tensor, sample_rate = torchaudio.load(audio_path)
        if sample_rate != target_sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, target_sample_rate)
            audio_tensor = resampler(audio_tensor)
        if audio_tensor.shape[0] > 1:
            audio_tensor = audio_tensor.mean(dim=0, keepdim=True)
        return audio_tensor.squeeze(0)
    except Exception as exc:
        logger.warning("Failed to load audio %s: %s", audio_path, exc)
        return None



def load_jsonl_records(path: Path, limit: int | None = None, seed: int = 42) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    if limit is None:
        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        rng.shuffle(records)
        return records

    reservoir: list[dict[str, Any]] = []
    seen = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if len(reservoir) < limit:
                reservoir.append(record)
            else:
                idx = rng.randint(0, seen)
                if idx < limit:
                    reservoir[idx] = record
            seen += 1
    rng.shuffle(reservoir)
    return reservoir



def discover_jsonl_files(dataset_path: str) -> list[Path]:
    root = Path(dataset_path)
    if not root.exists():
        return []
    if root.is_file() and root.suffix == ".jsonl":
        return [root]
    return sorted(root.rglob("*.jsonl"))



def sample_dataset_records(spec: DatasetSpec, max_samples: int, seed: int) -> list[dict[str, Any]]:
    jsonl_files = discover_jsonl_files(spec.path)
    if not jsonl_files:
        raise DataAvailabilityError(f"No jsonl files found for dataset {spec.name}: {spec.path}")

    records: list[dict[str, Any]] = []
    remaining = max_samples
    per_file_seed = seed
    for jsonl_file in jsonl_files:
        file_limit = remaining if remaining > 0 else None
        file_records = load_jsonl_records(jsonl_file, limit=file_limit, seed=per_file_seed)
        for record in file_records:
            record["_source_jsonl"] = str(jsonl_file)
            record["_dataset_name"] = spec.name
            record["_modality"] = spec.modality
            records.append(record)
            if len(records) >= max_samples:
                return records
        remaining = max_samples - len(records)
        per_file_seed += 1
        if remaining <= 0:
            break
    return records



def build_audio_sample(record: dict[str, Any], processor: AutoProcessor, max_seq_length: int, audio_sample_rate: int) -> dict[str, Any] | None:
    messages = normalize_messages(record.get("messages", []))
    source_jsonl = record.get("_source_jsonl", "")
    base_dir = str(Path(source_jsonl).parent)
    audio_paths = record.get("audios") or _extract_audio_paths_from_messages(record.get("messages", []), base_dir)
    audio_path = audio_paths[0] if audio_paths else None
    audio_tensor = load_audio_tensor(audio_path, audio_sample_rate) if audio_path else None
    if audio_tensor is None:
        return None

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    model_inputs = processor(
        text=text,
        audio=audio_tensor,
        sampling_rate=audio_sample_rate,
        return_tensors="pt",
    )
    if "input_ids" in model_inputs and model_inputs["input_ids"].shape[-1] > max_seq_length:
        for key, value in list(model_inputs.items()):
            if isinstance(value, torch.Tensor) and value.dim() >= 2 and value.shape[-1] == model_inputs["input_ids"].shape[-1]:
                model_inputs[key] = value[..., :max_seq_length]

    return {
        "dataset": record.get("_dataset_name"),
        "modality": "audio",
        "messages": messages,
        "inputs": model_inputs,
    }



def build_text_sample(record: dict[str, Any], processor: AutoProcessor, max_seq_length: int) -> dict[str, Any] | None:
    messages = normalize_messages(record.get("messages", []))
    if not messages:
        return None
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    model_inputs = processor.tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_seq_length,
    )
    return {
        "dataset": record.get("_dataset_name"),
        "modality": "text",
        "messages": messages,
        "inputs": model_inputs,
    }



def load_modality_samples(
    processor: AutoProcessor,
    modality: str,
    max_datasets: int | None,
    max_samples_per_dataset: int,
    sample_seed: int,
    max_seq_length: int,
    audio_sample_rate: int,
    dataset_root: str | None = None,
    total_samples: int | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    samples: list[dict[str, Any]] = []
    warnings: list[str] = []
    rng = random.Random(sample_seed)
    selected_specs = [spec for spec in build_dataset_specs(dataset_root) if spec.modality == modality]
    if max_datasets is not None:
        rng.shuffle(selected_specs)
        selected_specs = selected_specs[:max_datasets]

    if total_samples is not None and selected_specs:
        base = max(1, total_samples // len(selected_specs))
        per_dataset_limits = {spec.name: base for spec in selected_specs}
        remainder = max(0, total_samples - base * len(selected_specs))
        for spec in selected_specs[:remainder]:
            per_dataset_limits[spec.name] += 1
    else:
        per_dataset_limits = {spec.name: max_samples_per_dataset for spec in selected_specs}

    for idx, spec in enumerate(selected_specs):
        try:
            records = sample_dataset_records(spec, per_dataset_limits[spec.name], sample_seed + idx)
        except DataAvailabilityError as exc:
            warnings.append(str(exc))
            continue

        for record in records:
            if modality == "audio":
                sample = build_audio_sample(record, processor, max_seq_length, audio_sample_rate)
            else:
                sample = build_text_sample(record, processor, max_seq_length)
            if sample is not None:
                samples.append(sample)

    rng.shuffle(samples)
    if total_samples is not None:
        samples = samples[:total_samples]
    return samples, warnings



def materialize_model_inputs(model, inputs: dict[str, Any]) -> dict[str, Any]:
    device = next(model.thinker.parameters()).device
    model_dtype = next(model.thinker.parameters()).dtype
    prepared: dict[str, Any] = {}
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            value = value.to(device)
            if value.is_floating_point():
                value = value.to(model_dtype)
        prepared[key] = value
    return prepared
