from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from modality_reap.args import ClusterArgs, DataArgs, MergeArgs, ModelArgs, ObserverArgs, ReapArgs, ReportArgs, ensure_output_dir
from modality_reap.eval import (
    evaluate_generation,
    evaluate_teacher_forcing,
    load_eval_samples,
    save_evaluation_results,
    select_generation_subset,
)
from modality_reap.main import (
    collect_observation_sets,
    load_model_and_processors,
    prepare_modality_samples,
    run_compression_pipeline,
)
from modality_reap.model_util import MODEL_ATTRS, get_moe
from modality_reap.reporting import save_json

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "requirement4_20260407"

ABLATION_VARIANTS: list[dict[str, Any]] = [
    {
        "name": "baseline_reap",
        "description": "关闭三项创新，作为传统统一压缩基线。",
        "cluster_overrides": {
            "use_hybrid_strategy": False,
            "use_layer_adaptive_schedule": False,
        },
        "merge_overrides": {
            "merge_method": "frequency_weighted_average",
            "dom_as_base": False,
        },
    },
    {
        "name": "hybrid_only",
        "description": "仅启用 Hybrid Modality-REAP。",
        "cluster_overrides": {
            "use_hybrid_strategy": True,
            "use_layer_adaptive_schedule": False,
        },
        "merge_overrides": {
            "merge_method": "frequency_weighted_average",
            "dom_as_base": False,
        },
    },
    {
        "name": "hybrid_plus_schedule",
        "description": "启用 Hybrid + Layer-Adaptive Compression Schedule。",
        "cluster_overrides": {
            "use_hybrid_strategy": True,
            "use_layer_adaptive_schedule": True,
        },
        "merge_overrides": {
            "merge_method": "frequency_weighted_average",
            "dom_as_base": False,
        },
    },
    {
        "name": "full_modality_reap",
        "description": "启用三项创新：Hybrid + Layer-Adaptive + conflict-aware subspace merge。",
        "cluster_overrides": {
            "use_hybrid_strategy": True,
            "use_layer_adaptive_schedule": True,
        },
        "merge_overrides": {
            "merge_method": "conflict_aware_subspace",
            "dom_as_base": True,
        },
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ModalityReap requirement4 experiments.")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--model-name", type=str, default="/data/szs/share/Qwen3-Omni-30B-A3B-Instruct")
    parser.add_argument("--attn-implementation", type=str, default="sdpa")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compression-ratio", type=float, default=0.5)
    parser.add_argument("--obs-audio-samples", type=int, default=20)
    parser.add_argument("--obs-text-samples", type=int, default=9)
    parser.add_argument("--eval-audio-per-dataset", type=int, default=3)
    parser.add_argument("--eval-text-per-dataset", type=int, default=2)
    parser.add_argument("--gen-audio-per-dataset", type=int, default=1)
    parser.add_argument("--gen-text-per-dataset", type=int, default=1)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--audio-sample-rate", type=int, default=16000)
    parser.add_argument("--save-full-model", action="store_true")
    return parser.parse_args()


def cleanup_model(model) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def flush_cuda_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def snapshot_moe_state(model) -> dict[str, Any]:
    model_attrs = MODEL_ATTRS[model.__class__.__name__]
    gate_proj_name = model_attrs["gate_proj"]
    down_proj_name = model_attrs["down_proj"]
    text_config = model.config.thinker_config.text_config
    snapshot: dict[str, Any] = {
        "layers": {},
        "config_num_experts": int(text_config.num_experts),
        "config_num_experts_per_layer": list(getattr(text_config, "num_experts_per_layer", [])) or None,
        "thinker_num_experts": int(getattr(model.thinker, "num_experts", text_config.num_experts)),
        "thinker_num_experts_per_tok": int(getattr(model.thinker, "num_experts_per_tok", text_config.num_experts_per_tok)),
    }
    num_layers = len(model.thinker.model.layers)
    for layer_idx in range(num_layers):
        moe = get_moe(model, layer_idx)
        if not hasattr(moe, "experts") or not hasattr(moe, "gate"):
            continue
        experts = moe.experts
        gate = moe.gate
        snapshot["layers"][layer_idx] = {
            "gate_up_proj": getattr(experts, gate_proj_name).detach().cpu().clone(),
            "down_proj": getattr(experts, down_proj_name).detach().cpu().clone(),
            "experts_num_experts": int(experts.num_experts),
            "gate_weight": gate.weight.detach().cpu().clone(),
            "gate_num_experts": int(gate.num_experts),
            "gate_top_k": int(getattr(gate, "top_k", 0)),
        }
    return snapshot


def restore_moe_state(model, snapshot: dict[str, Any]) -> None:
    model_attrs = MODEL_ATTRS[model.__class__.__name__]
    gate_proj_name = model_attrs["gate_proj"]
    down_proj_name = model_attrs["down_proj"]
    for layer_idx, layer_state in snapshot["layers"].items():
        moe = get_moe(model, layer_idx)
        experts = moe.experts
        gate = moe.gate

        gate_proj = getattr(experts, gate_proj_name)
        down_proj = getattr(experts, down_proj_name)
        gate_weight = gate.weight

        setattr(
            experts,
            gate_proj_name,
            torch.nn.Parameter(layer_state["gate_up_proj"].to(device=gate_proj.device, dtype=gate_proj.dtype).contiguous()),
        )
        setattr(
            experts,
            down_proj_name,
            torch.nn.Parameter(layer_state["down_proj"].to(device=down_proj.device, dtype=down_proj.dtype).contiguous()),
        )
        experts.num_experts = int(layer_state["experts_num_experts"])

        gate.weight = torch.nn.Parameter(
            layer_state["gate_weight"].to(device=gate_weight.device, dtype=gate_weight.dtype).contiguous()
        )
        gate.num_experts = int(layer_state["gate_num_experts"])
        if hasattr(gate, "top_k") and int(layer_state["gate_top_k"]) > 0:
            gate.top_k = int(layer_state["gate_top_k"])

    text_config = model.config.thinker_config.text_config
    text_config.num_experts = int(snapshot["config_num_experts"])
    if snapshot["config_num_experts_per_layer"] is None:
        if hasattr(text_config, "num_experts_per_layer"):
            delattr(text_config, "num_experts_per_layer")
    else:
        setattr(text_config, "num_experts_per_layer", list(snapshot["config_num_experts_per_layer"]))
    if hasattr(model.thinker, "num_experts"):
        model.thinker.num_experts = int(snapshot["thinker_num_experts"])
    if hasattr(model.thinker, "num_experts_per_tok"):
        model.thinker.num_experts_per_tok = int(snapshot["thinker_num_experts_per_tok"])
    flush_cuda_cache()


def build_compression_summary(compression_result) -> dict[str, Any]:
    total_before = sum(plan.num_experts for plan in compression_result.compression_plans.values())
    total_target = sum(plan.target_experts for plan in compression_result.compression_plans.values())
    total_after = sum(compression_result.layer_num_experts.values())
    total_keep = sum(len(plan.keep_experts) for plan in compression_result.compression_plans.values())
    total_merge = sum(len(plan.merge_experts) for plan in compression_result.compression_plans.values())
    total_prune = sum(len(plan.pruned_experts) for plan in compression_result.compression_plans.values())
    per_layer_target_hits = sum(
        int(compression_result.layer_num_experts[layer_idx] == plan.target_experts)
        for layer_idx, plan in compression_result.compression_plans.items()
    )
    layer_count = max(len(compression_result.compression_plans), 1)
    achieved_ratio = 1.0 - (total_after / max(total_before, 1))
    target_ratio = 1.0 - (total_target / max(total_before, 1))
    return {
        "total_experts_before": total_before,
        "total_experts_target": total_target,
        "total_experts_after": total_after,
        "achieved_compression_ratio": achieved_ratio,
        "target_compression_ratio": target_ratio,
        "total_keep_experts": total_keep,
        "total_merge_candidates": total_merge,
        "total_pruned_experts": total_prune,
        "per_layer_target_hit_rate": per_layer_target_hits / layer_count,
        "mean_experts_per_layer_after": total_after / layer_count,
    }


def build_variant_summary(
    *,
    variant: dict[str, Any],
    teacher_forcing: dict[str, Any],
    generation: dict[str, Any],
    compression_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    overall_tf = teacher_forcing["aggregate"].get("overall", {})
    overall_gen = generation["aggregate"].get("overall", {})
    summary = {
        "name": variant["name"],
        "description": variant["description"],
        "teacher_forcing": overall_tf,
        "generation": overall_gen,
        "compression": compression_summary,
    }
    if compression_summary is not None:
        summary["verification"] = {
            "real_model_flow_passed": True,
            "conflict_aware_subspace_enabled": variant["merge_overrides"]["merge_method"] == "conflict_aware_subspace",
            "compression_plan_effective": compression_summary["total_experts_after"] < compression_summary["total_experts_before"],
        }
    else:
        summary["verification"] = {
            "real_model_flow_passed": True,
            "conflict_aware_subspace_enabled": False,
            "compression_plan_effective": False,
        }
    return summary


def write_summary_markdown(output_dir: Path, leaderboard: list[dict[str, Any]]) -> None:
    lines = [
        "# Requirement 4 Summary",
        "",
        f"- Generated at: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "| Variant | Compression | TF Loss | TF PPL | TF Token Acc | Gen Rouge-L | Gen Token F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in leaderboard:
        compression = row.get("compression_ratio", 0.0)
        tf_loss = row.get("tf_loss", 0.0)
        tf_ppl = row.get("tf_ppl", 0.0)
        tf_acc = row.get("tf_token_accuracy", 0.0)
        rouge_l = row.get("gen_rouge_l_f1", 0.0)
        token_f1 = row.get("gen_token_f1", 0.0)
        lines.append(
            f"| {row['name']} | {compression:.4f} | {tf_loss:.4f} | {tf_ppl:.4f} | {tf_acc:.4f} | {rouge_l:.4f} | {token_f1:.4f} |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = ensure_output_dir(args.output_dir)
    observation_dir = ensure_output_dir(output_dir / "observation_cache")
    baseline_dir = ensure_output_dir(output_dir / "original_model")
    observation_cache_path = observation_dir / "observations.pt"
    observation_manifest_path = observation_dir / "manifest.json"
    baseline_summary_path = baseline_dir / "summary.json"
    baseline_eval_dir = baseline_dir / "evaluation"

    experiment_config = {
        "model_name": args.model_name,
        "seed": args.seed,
        "compression_ratio": args.compression_ratio,
        "obs_audio_samples": args.obs_audio_samples,
        "obs_text_samples": args.obs_text_samples,
        "eval_audio_per_dataset": args.eval_audio_per_dataset,
        "eval_text_per_dataset": args.eval_text_per_dataset,
        "gen_audio_per_dataset": args.gen_audio_per_dataset,
        "gen_text_per_dataset": args.gen_text_per_dataset,
        "max_seq_length": args.max_seq_length,
        "max_new_tokens": args.max_new_tokens,
    }
    save_json(output_dir / "experiment_config.json", experiment_config)

    base_reap_args = ReapArgs(seed=args.seed, profile=False, do_save=False, smoke_test=False)
    base_model_args = ModelArgs(
        model_name=args.model_name,
        attn_implementation=args.attn_implementation,
        disable_talker=True,
    )
    observation_data_args = DataArgs(
        total_audio_samples=args.obs_audio_samples,
        total_text_samples=args.obs_text_samples,
        max_seq_length=args.max_seq_length,
        audio_sample_rate=args.audio_sample_rate,
        output_dir=str(observation_dir),
    )
    base_obs_args = ObserverArgs()
    base_cluster_args = ClusterArgs(compression_ratio=args.compression_ratio)
    base_merge_args = MergeArgs()
    base_report_args = ReportArgs()

    logger.info("加载原始模型")
    original_model, processor, tokenizer = load_model_and_processors(base_model_args)
    if observation_cache_path.exists():
        logger.info("复用已有 observation cache: %s", observation_cache_path)
        observation_sets = torch.load(observation_cache_path, map_location="cpu")
    else:
        logger.info("收集 observation cache")
        audio_samples, text_samples, observation_warnings = prepare_modality_samples(processor, observation_data_args)
        observation_sets = collect_observation_sets(original_model, audio_samples, text_samples, base_obs_args)
        torch.save(observation_sets, observation_cache_path)
        save_json(
            observation_manifest_path,
            {
                "warnings": observation_warnings,
                "audio_samples": len(audio_samples),
                "text_samples": len(text_samples),
            },
        )

    logger.info("加载统一评测样本")
    eval_samples = load_eval_samples(
        audio_samples_per_dataset=args.eval_audio_per_dataset,
        text_samples_per_dataset=args.eval_text_per_dataset,
        sample_seed=args.seed + 100,
        audio_sample_rate=args.audio_sample_rate,
    )
    generation_samples = select_generation_subset(
        eval_samples,
        audio_per_dataset=args.gen_audio_per_dataset,
        text_per_dataset=args.gen_text_per_dataset,
    )
    save_json(
        output_dir / "evaluation_manifest.json",
        {
            "total_eval_samples": len(eval_samples),
            "generation_subset_samples": len(generation_samples),
        },
    )

    if baseline_summary_path.exists() and (baseline_eval_dir / "teacher_forcing_summary.json").exists():
        logger.info("复用已有原始模型评测结果: %s", baseline_summary_path)
        baseline_summary = json.loads(baseline_summary_path.read_text(encoding="utf-8"))
    else:
        logger.info("评测原始模型")
        baseline_teacher_forcing = evaluate_teacher_forcing(
            original_model,
            processor,
            eval_samples,
            max_seq_length=args.max_seq_length,
        )
        baseline_generation = evaluate_generation(
            original_model,
            processor,
            generation_samples,
            max_seq_length=args.max_seq_length,
            max_new_tokens=args.max_new_tokens,
        )
        save_evaluation_results(
            baseline_eval_dir,
            teacher_forcing=baseline_teacher_forcing,
            generation=baseline_generation,
            sample_manifest=eval_samples,
            generation_manifest=generation_samples,
        )
        baseline_summary = build_variant_summary(
            variant={
                "name": "original_model",
                "description": "未压缩原模型。",
                "merge_overrides": {"merge_method": "none"},
            },
            teacher_forcing=baseline_teacher_forcing,
            generation=baseline_generation,
            compression_summary=None,
        )
        save_json(baseline_summary_path, baseline_summary)

    base_moe_snapshot = snapshot_moe_state(original_model)

    leaderboard: list[dict[str, Any]] = [
        {
            "name": "original_model",
            "compression_ratio": 0.0,
            "tf_loss": baseline_summary["teacher_forcing"].get("loss", 0.0),
            "tf_ppl": baseline_summary["teacher_forcing"].get("ppl", 0.0),
            "tf_token_accuracy": baseline_summary["teacher_forcing"].get("token_accuracy", 0.0),
            "gen_rouge_l_f1": baseline_summary["generation"].get("best_rouge_l_f1", 0.0),
            "gen_token_f1": baseline_summary["generation"].get("best_token_f1", 0.0),
        }
    ]

    for variant in ABLATION_VARIANTS:
        logger.info("运行变体: %s", variant["name"])
        variant_dir = ensure_output_dir(output_dir / variant["name"])
        variant_summary_path = variant_dir / "summary.json"
        if variant_summary_path.exists():
            logger.info("复用已存在变体结果: %s", variant_summary_path)
            summary = json.loads(variant_summary_path.read_text(encoding="utf-8"))
            compression_summary = summary.get("compression") or {}
            leaderboard.append(
                {
                    "name": variant["name"],
                    "compression_ratio": compression_summary.get("achieved_compression_ratio", 0.0),
                    "tf_loss": summary["teacher_forcing"].get("loss", 0.0),
                    "tf_ppl": summary["teacher_forcing"].get("ppl", 0.0),
                    "tf_token_accuracy": summary["teacher_forcing"].get("token_accuracy", 0.0),
                    "gen_rouge_l_f1": summary["generation"].get("best_rouge_l_f1", 0.0),
                    "gen_token_f1": summary["generation"].get("best_token_f1", 0.0),
                }
            )
            continue

        restore_moe_state(original_model, base_moe_snapshot)
        variant_reap_args = dataclasses.replace(
            base_reap_args,
            do_save=args.save_full_model and variant["name"] == "full_modality_reap",
        )
        variant_data_args = dataclasses.replace(
            observation_data_args,
            output_dir=str(variant_dir / "compression"),
        )
        variant_cluster_args = dataclasses.replace(base_cluster_args, **variant["cluster_overrides"])
        variant_merge_args = dataclasses.replace(base_merge_args, **variant["merge_overrides"])

        compression_result = run_compression_pipeline(
            reap_args=variant_reap_args,
            model_args=base_model_args,
            data_args=variant_data_args,
            obs_args=base_obs_args,
            cluster_args=variant_cluster_args,
            merge_args=variant_merge_args,
            report_args=base_report_args,
            model=original_model,
            processor=processor,
            tokenizer=tokenizer,
            observation_sets=observation_sets,
            save_observation_artifacts=False,
            observation_artifact_reference=str(observation_dir),
            validate_merge=False,
        )
        teacher_forcing = evaluate_teacher_forcing(
            compression_result.model,
            processor,
            eval_samples,
            max_seq_length=args.max_seq_length,
        )
        generation = evaluate_generation(
            compression_result.model,
            processor,
            generation_samples,
            max_seq_length=args.max_seq_length,
            max_new_tokens=args.max_new_tokens,
        )
        save_evaluation_results(
            variant_dir / "evaluation",
            teacher_forcing=teacher_forcing,
            generation=generation,
            sample_manifest=eval_samples,
            generation_manifest=generation_samples,
        )
        compression_summary = build_compression_summary(compression_result)
        summary = build_variant_summary(
            variant=variant,
            teacher_forcing=teacher_forcing,
            generation=generation,
            compression_summary=compression_summary,
        )
        save_json(variant_dir / "summary.json", summary)
        leaderboard.append(
            {
                "name": variant["name"],
                "compression_ratio": compression_summary["achieved_compression_ratio"],
                "tf_loss": summary["teacher_forcing"].get("loss", 0.0),
                "tf_ppl": summary["teacher_forcing"].get("ppl", 0.0),
                "tf_token_accuracy": summary["teacher_forcing"].get("token_accuracy", 0.0),
                "gen_rouge_l_f1": summary["generation"].get("best_rouge_l_f1", 0.0),
                "gen_token_f1": summary["generation"].get("best_token_f1", 0.0),
            }
        )
        flush_cuda_cache()

    save_json(output_dir / "leaderboard.json", {"rows": leaderboard})
    write_summary_markdown(output_dir, leaderboard)
    logger.info("Requirement 4 实验完成，输出目录: %s", output_dir)
    cleanup_model(original_model)


if __name__ == "__main__":
    main()
