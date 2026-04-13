from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

from modality_reap.data import DatasetSpec, load_audio_tensor, load_jsonl_records, materialize_model_inputs, normalize_messages
from modality_reap.reporting import save_json


DEFAULT_AUDIO_EVAL_SPECS: list[DatasetSpec] = [
    DatasetSpec(
        "Clotho-validation",
        "audio",
        "/data/szs/share/voice_model_project/datasets/Clotho/Clotho_validation_AudioCaptioning.jsonl",
    ),
    DatasetSpec(
        "AudioCaps-test",
        "audio",
        "/data/szs/share/voice_model_project/datasets/AudioCaps/AudioCaps_test_AudioCaptioning.jsonl",
    ),
]

DEFAULT_TEXT_EVAL_SPECS: list[DatasetSpec] = [
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


@dataclass(frozen=True)
class EvalSample:
    sample_id: str
    dataset: str
    modality: str
    prompt_messages: list[dict[str, str]]
    full_messages: list[dict[str, str]]
    target_text: str
    references: list[str]
    audio_tensor: torch.Tensor | None = None
    audio_sample_rate: int = 16000

    def to_manifest(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["audio_tensor"] = None
        return payload


def _resolve_audio_path(audio_path: str, base_path: Path) -> str:
    if not audio_path:
        return audio_path
    if Path(audio_path).is_absolute():
        return audio_path
    return str((base_path / audio_path).resolve())


def _extract_audio_paths(messages: list[dict[str, Any]], base_path: Path) -> list[str]:
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


def _build_eval_sample(
    record: dict[str, Any],
    *,
    dataset_name: str,
    modality: str,
    source_path: Path,
    audio_sample_rate: int,
) -> EvalSample | None:
    messages = record.get("messages", [])
    if len(messages) < 2:
        return None

    normalized_messages = normalize_messages(messages)
    if len(normalized_messages) < 2:
        return None
    if normalized_messages[-1]["role"] != "assistant":
        return None

    prompt_messages = normalized_messages[:-1]
    target_text = normalized_messages[-1]["content"].strip()
    if not prompt_messages or not target_text:
        return None

    audio_tensor = None
    if modality == "audio":
        base_path = source_path.parent
        audio_paths = record.get("audios") or _extract_audio_paths(messages, base_path)
        if not audio_paths:
            return None
        audio_tensor = load_audio_tensor(audio_paths[0], audio_sample_rate)
        if audio_tensor is None:
            return None

    references = [
        str(reference).strip()
        for reference in record.get("references", [target_text])
        if str(reference).strip()
    ]
    if not references:
        references = [target_text]

    return EvalSample(
        sample_id=str(record.get("id", f"{dataset_name}-{record.get('caption_index', 'sample')}")),
        dataset=dataset_name,
        modality=modality,
        prompt_messages=prompt_messages,
        full_messages=normalized_messages,
        target_text=target_text,
        references=references,
        audio_tensor=audio_tensor,
        audio_sample_rate=audio_sample_rate,
    )


def load_eval_samples(
    *,
    audio_samples_per_dataset: int,
    text_samples_per_dataset: int,
    sample_seed: int,
    audio_sample_rate: int,
) -> list[EvalSample]:
    samples: list[EvalSample] = []
    spec_seed = sample_seed
    for spec in DEFAULT_AUDIO_EVAL_SPECS:
        records = load_jsonl_records(Path(spec.path), limit=audio_samples_per_dataset, seed=spec_seed)
        spec_seed += 1
        for record in records:
            sample = _build_eval_sample(
                record,
                dataset_name=spec.name,
                modality=spec.modality,
                source_path=Path(spec.path),
                audio_sample_rate=audio_sample_rate,
            )
            if sample is not None:
                samples.append(sample)

    for spec in DEFAULT_TEXT_EVAL_SPECS:
        records = load_jsonl_records(Path(spec.path), limit=text_samples_per_dataset, seed=spec_seed)
        spec_seed += 1
        for record in records:
            sample = _build_eval_sample(
                record,
                dataset_name=spec.name,
                modality=spec.modality,
                source_path=Path(spec.path),
                audio_sample_rate=audio_sample_rate,
            )
            if sample is not None:
                samples.append(sample)
    return samples


def select_generation_subset(
    samples: list[EvalSample],
    *,
    audio_per_dataset: int,
    text_per_dataset: int,
) -> list[EvalSample]:
    per_dataset_counts: dict[tuple[str, str], int] = defaultdict(int)
    selected: list[EvalSample] = []
    for sample in samples:
        limit = audio_per_dataset if sample.modality == "audio" else text_per_dataset
        key = (sample.modality, sample.dataset)
        if per_dataset_counts[key] >= limit:
            continue
        selected.append(sample)
        per_dataset_counts[key] += 1
    return selected


def _truncate_multimodal_inputs(inputs: dict[str, Any], max_seq_length: int) -> dict[str, Any]:
    if "input_ids" not in inputs or inputs["input_ids"].shape[-1] <= max_seq_length:
        return inputs
    seq_len = inputs["input_ids"].shape[-1]
    for key, value in list(inputs.items()):
        if isinstance(value, torch.Tensor) and value.dim() >= 2 and value.shape[-1] == seq_len:
            inputs[key] = value[..., :max_seq_length]
    return inputs


def _build_inputs(
    processor,
    messages: list[dict[str, str]],
    *,
    audio_tensor: torch.Tensor | None,
    audio_sample_rate: int,
    max_seq_length: int,
    add_generation_prompt: bool,
) -> dict[str, Any]:
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
    if audio_tensor is None:
        return processor.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_seq_length,
        )
    inputs = processor(
        text=text,
        audio=audio_tensor,
        sampling_rate=audio_sample_rate,
        return_tensors="pt",
    )
    return _truncate_multimodal_inputs(inputs, max_seq_length)


def prepare_labeled_inputs(
    processor,
    sample: EvalSample,
    *,
    max_seq_length: int,
) -> tuple[dict[str, Any], int] | None:
    prompt_inputs = _build_inputs(
        processor,
        sample.prompt_messages,
        audio_tensor=sample.audio_tensor,
        audio_sample_rate=sample.audio_sample_rate,
        max_seq_length=max_seq_length,
        add_generation_prompt=True,
    )
    full_inputs = _build_inputs(
        processor,
        sample.full_messages,
        audio_tensor=sample.audio_tensor,
        audio_sample_rate=sample.audio_sample_rate,
        max_seq_length=max_seq_length,
        add_generation_prompt=False,
    )
    prompt_len = min(prompt_inputs["input_ids"].shape[1], full_inputs["input_ids"].shape[1])
    labels = full_inputs["input_ids"].clone()
    labels[:, :prompt_len] = -100
    if "attention_mask" in full_inputs:
        labels[full_inputs["attention_mask"] == 0] = -100
    valid_target_tokens = int((labels != -100).sum().item())
    if valid_target_tokens <= 0:
        return None
    full_inputs["labels"] = labels
    return full_inputs, valid_target_tokens


def _safe_exp(loss_value: float) -> float:
    return float(math.exp(min(loss_value, 20.0)))


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", _normalize_text(text), flags=re.UNICODE)


def _lcs_length(left: list[str], right: list[str]) -> int:
    if not left or not right:
        return 0
    table = [0] * (len(right) + 1)
    for left_token in left:
        prev = 0
        for idx, right_token in enumerate(right, start=1):
            current = table[idx]
            if left_token == right_token:
                table[idx] = prev + 1
            else:
                table[idx] = max(table[idx], table[idx - 1])
            prev = current
    return table[-1]


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    remaining = list(ref_tokens)
    overlap = 0
    for token in pred_tokens:
        if token in remaining:
            remaining.remove(token)
            overlap += 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def rouge_l_f1(prediction: str, reference: str) -> float:
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = _lcs_length(pred_tokens, ref_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _best_reference_metric(
    prediction: str,
    references: list[str],
    metric_fn,
) -> tuple[float, str]:
    best_score = -1.0
    best_reference = references[0] if references else ""
    for reference in references:
        score = float(metric_fn(prediction, reference))
        if score > best_score:
            best_score = score
            best_reference = reference
    return best_score, best_reference


def exact_match(prediction: str, reference: str) -> float:
    return float(_normalize_text(prediction) == _normalize_text(reference))


def _decode_generated_text(processor, sequences: torch.Tensor, prompt_len: int) -> str:
    decoded = processor.batch_decode(
        sequences[:, prompt_len:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return decoded[0].strip()


def _grouped_metric_summary(records: list[dict[str, Any]], metric_keys: list[str]) -> dict[str, Any]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        for group_name in ("overall", f"modality:{record['modality']}", f"dataset:{record['dataset']}"):
            grouped[group_name]["samples"].append(1.0)
            for key in metric_keys:
                grouped[group_name][key].append(float(record[key]))
            if "target_tokens" in record:
                grouped[group_name]["target_tokens"].append(float(record["target_tokens"]))

    summary: dict[str, Any] = {}
    for group_name, metrics in grouped.items():
        entry: dict[str, Any] = {
            "samples": int(sum(metrics["samples"])),
        }
        if "target_tokens" in metrics:
            entry["target_tokens"] = int(sum(metrics["target_tokens"]))
        for key in metric_keys:
            values = metrics[key]
            entry[key] = float(sum(values) / max(len(values), 1))
        summary[group_name] = entry
    return summary


def evaluate_teacher_forcing(
    model,
    processor,
    samples: list[EvalSample],
    *,
    max_seq_length: int,
) -> dict[str, Any]:
    sample_results: list[dict[str, Any]] = []
    teacher_model = model.thinker if hasattr(model, "thinker") else model
    for sample in tqdm(samples, desc="Teacher-forcing eval", unit="sample"):
        prepared = prepare_labeled_inputs(
            processor,
            sample,
            max_seq_length=max_seq_length,
        )
        if prepared is None:
            continue
        inputs, valid_target_tokens = prepared
        model_inputs = materialize_model_inputs(model, inputs)
        with torch.no_grad():
            outputs = teacher_model(**model_inputs)

        logits = outputs.logits[:, :-1].detach()
        target_ids = model_inputs["input_ids"][:, 1:]
        label_mask = model_inputs["labels"][:, 1:] != -100
        total_tokens = int(label_mask.sum().item())
        correct_tokens = int(((logits.argmax(dim=-1) == target_ids) & label_mask).sum().item())
        token_accuracy = correct_tokens / max(total_tokens, 1)
        loss_value = float(outputs.loss.item())
        sample_results.append(
            {
                "sample_id": sample.sample_id,
                "dataset": sample.dataset,
                "modality": sample.modality,
                "loss": loss_value,
                "ppl": _safe_exp(loss_value),
                "token_accuracy": token_accuracy,
                "target_tokens": valid_target_tokens,
            }
        )

    return {
        "aggregate": _grouped_metric_summary(sample_results, ["loss", "ppl", "token_accuracy"]),
        "samples": sample_results,
    }


def evaluate_generation(
    model,
    processor,
    samples: list[EvalSample],
    *,
    max_seq_length: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    sample_results: list[dict[str, Any]] = []
    for sample in tqdm(samples, desc="Generation eval", unit="sample"):
        prompt_inputs = _build_inputs(
            processor,
            sample.prompt_messages,
            audio_tensor=sample.audio_tensor,
            audio_sample_rate=sample.audio_sample_rate,
            max_seq_length=max_seq_length,
            add_generation_prompt=True,
        )
        prompt_len = int(prompt_inputs["input_ids"].shape[1])
        model_inputs = materialize_model_inputs(model, prompt_inputs)

        with torch.no_grad():
            generation = model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                return_audio=False,
                thinker_return_dict_in_generate=True,
            )

        if isinstance(generation, tuple):
            text_generation = generation[0]
        else:
            text_generation = generation
        sequences = text_generation.sequences if hasattr(text_generation, "sequences") else text_generation
        prediction = _decode_generated_text(processor, sequences, prompt_len)

        best_exact, best_exact_ref = _best_reference_metric(prediction, sample.references, exact_match)
        best_token_f1, best_token_ref = _best_reference_metric(prediction, sample.references, token_f1)
        best_rouge_l, best_rouge_ref = _best_reference_metric(prediction, sample.references, rouge_l_f1)

        sample_results.append(
            {
                "sample_id": sample.sample_id,
                "dataset": sample.dataset,
                "modality": sample.modality,
                "prediction": prediction,
                "target_text": sample.target_text,
                "best_exact_match": best_exact,
                "best_exact_match_reference": best_exact_ref,
                "best_token_f1": best_token_f1,
                "best_token_f1_reference": best_token_ref,
                "best_rouge_l_f1": best_rouge_l,
                "best_rouge_l_reference": best_rouge_ref,
            }
        )

    return {
        "aggregate": _grouped_metric_summary(
            sample_results,
            ["best_exact_match", "best_token_f1", "best_rouge_l_f1"],
        ),
        "samples": sample_results,
    }


def save_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_evaluation_results(
    output_dir: Path,
    *,
    teacher_forcing: dict[str, Any],
    generation: dict[str, Any],
    sample_manifest: list[EvalSample],
    generation_manifest: list[EvalSample],
) -> None:
    save_json(output_dir / "sample_manifest.json", {"samples": [sample.to_manifest() for sample in sample_manifest]})
    save_json(
        output_dir / "generation_manifest.json",
        {"samples": [sample.to_manifest() for sample in generation_manifest]},
    )
    save_json(output_dir / "teacher_forcing_summary.json", teacher_forcing["aggregate"])
    save_json(output_dir / "generation_summary.json", generation["aggregate"])
    save_jsonl(output_dir / "teacher_forcing_samples.jsonl", teacher_forcing["samples"])
    save_jsonl(output_dir / "generation_samples.jsonl", generation["samples"])
