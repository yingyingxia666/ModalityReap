from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ReapArgs:
    seed: int = field(default=42)
    debug: bool = field(default=False)
    profile: bool = field(default=True)
    run_observer_only: bool = field(default=False)
    do_save: bool = field(default=True)
    smoke_test: bool = field(default=True)


@dataclass
class ModelArgs:
    model_name: str = field(default="/data/szs/share/Qwen3-Omni-30B-A3B-Instruct")
    attn_implementation: str = field(default="sdpa")


@dataclass
class DataArgs:
    audio_ratio: float = field(default=0.75)
    text_ratio: float = field(default=0.25)
    total_audio_samples: int = field(default=200)
    total_text_samples: int = field(default=50)
    max_audio_datasets: Optional[int] = field(default=None)
    max_text_datasets: Optional[int] = field(default=None)
    max_samples_per_dataset: int = field(default=50)
    sample_randomly: bool = field(default=True)
    sample_seed: int = field(default=42)
    max_seq_length: int = field(default=2048)
    audio_sample_rate: int = field(default=16000)
    dataset_root: str = field(default="/data/szs/share/voice_model_project/datasets")
    output_dir: str = field(default="./artifacts/modality_reap")


@dataclass
class ObserverArgs:
    overwrite_observations: bool = field(default=False)
    renormalize_router_weights: bool = field(default=False)
    record_pruning_metrics_only: bool = field(default=False)
    output_file_name: str = field(default="observations.pt")


@dataclass
class ClusterArgs:
    compression_ratio: float = field(default=0.5)
    num_clusters: Optional[int] = field(default=None)
    cluster_method: str = field(default="agglomerative")
    linkage_method: str = field(default="average")
    frequency_penalty: bool = field(default=True)
    softmax_temperature: Optional[float] = field(default=None)
    max_cluster_size: Optional[int] = field(default=None)
    audio_protect_ratio: float = field(default=0.1)
    protect_middle_layers: bool = field(default=True)
    middle_layer_multiplier: float = field(default=1.5)
    use_router_analysis_prior: bool = field(default=True)


@dataclass
class MergeArgs:
    merge_method: str = field(default="frequency_weighted_average")
    permute: Optional[str] = field(default=None)
    dom_as_base: bool = field(default=False)
    select_top_k: float = field(default=0.1)
    conservative_anchor_weight: float = field(default=0.9)
    save_as_tied_params: bool = field(default=False)
    overwrite_merged_model: bool = field(default=False)


@dataclass
class ReportArgs:
    save_json: bool = field(default=True)
    save_per_layer_csv: bool = field(default=False)


def ensure_output_dir(path: str | Path) -> Path:
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
