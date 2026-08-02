"""
extract_titan_concept_features.py

对指定任务数据集的 task-specific concept pool 使用 TITAN text encoder 提取文本特征。

默认输入:
  llm_filtered/<task_name>/final_task_concept_pool_patch_desc.csv
若该文件不存在，则回退到:
  llm_filtered/<task_name>/final_task_concept_pool.csv

示例:
  uv run python extract_titan_concept_features.py \
    --task_name CAMELYON16 \
    --batch_size 128

  python extract_titan_concept_features.py \
    --task_name TCGA-RCC \
    --include_patch_description \
    --prompt_mode pathology_templates \
    --batch_size 128 \
    --overwrite
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel


DEFAULT_DATASETS = ["CAMELYON16", "NSCLC", "TCGA-RCC", "UBC-OCEAN"]
DEFAULT_MODEL_NAME = "MahmoodLab/TITAN"
DESCRIPTION_COLUMN = "patch_visual_description_20x"
PATHOLOGY_PROMPT_TEMPLATES = [
    "a histopathology slide showing features of {text}",
    "microscopic analysis of {text}",
    "H&E stain clearly indicates {text}",
    "a region of interest featuring {text}",
    "tissue sample with presence of {text}",
    "cellular patterns characteristic of {text}",
    "pathological findings are consistent with {text}",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用 TITAN text encoder 为指定数据集的 concept pool 提取文本特征。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--task_name",
        choices=DEFAULT_DATASETS,
        default=None,
        help="需要处理的数据集名称。",
    )
    parser.add_argument(
        "--all_tasks",
        action="store_true",
        help="显式处理四个内置数据集；通常建议使用 --task_name 单独处理。",
    )
    parser.add_argument(
        "--input_root",
        default="llm_filtered",
        help="数据集 concept pool 的根目录。",
    )
    parser.add_argument(
        "--input_name",
        default=None,
        help=(
            "指定输入 CSV 文件名；不指定时优先使用 final_task_concept_pool_patch_desc.csv，"
            "不存在则使用 final_task_concept_pool.csv。"
        ),
    )
    parser.add_argument(
        "--output_dir_name",
        default="titan_features",
        help="每个数据集目录下保存 TITAN 特征的子目录名。",
    )
    parser.add_argument("--concept_column", default="concept", help="concept 文本列名。")
    parser.add_argument(
        "--description_column",
        default=DESCRIPTION_COLUMN,
        help="patch 视觉描述列名。",
    )
    parser.add_argument(
        "--include_patch_description",
        action="store_true",
        help="是否将 patch_visual_description_20x 与 concept 拼接后送入 TITAN。",
    )
    parser.add_argument(
        "--prompt_mode",
        choices=["none", "pathology_templates"],
        default="pathology_templates",
        help="是否使用病理 prompt templates；pathology_templates 会对多个模板编码后取平均。",
    )
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME, help="TITAN 模型名或本地路径。")
    parser.add_argument("--batch_size", type=int, default=128, help="TITAN 编码 batch size。")
    parser.add_argument(
        "--device",
        default=None,
        help="运行设备；不指定时自动选择 cuda 或 cpu。",
    )
    parser.add_argument(
        "--no_normalize",
        action="store_true",
        help="不对输出 embedding 做 L2 normalization。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="输出已存在时重新提取并覆盖。",
    )
    return parser.parse_args()


def resolve_input_path(input_root: Path, dataset: str, input_name: str | None) -> Path:
    dataset_dir = input_root / dataset
    if input_name:
        path = dataset_dir / input_name
        if not path.is_file():
            raise FileNotFoundError(f"找不到指定输入文件: {path}")
        return path

    preferred = dataset_dir / "final_task_concept_pool_patch_desc.csv"
    fallback = dataset_dir / "final_task_concept_pool.csv"
    if preferred.is_file():
        return preferred
    if fallback.is_file():
        return fallback
    raise FileNotFoundError(f"{dataset} 缺少输入 CSV: {preferred} 或 {fallback}")


def build_texts(
    df: pd.DataFrame,
    *,
    concept_column: str,
    description_column: str,
    include_patch_description: bool,
) -> Tuple[List[str], List[str]]:
    if concept_column not in df.columns:
        raise ValueError(f"CSV 缺少列: {concept_column}，现有列: {list(df.columns)}")

    concepts = df[concept_column].fillna("").astype(str).str.strip().tolist()
    if include_patch_description:
        if description_column not in df.columns:
            raise ValueError(
                f"--include_patch_description 需要列 {description_column}，"
                f"现有列: {list(df.columns)}"
            )
        descriptions = df[description_column].fillna("").astype(str).str.strip().tolist()
        texts = [
            f"{concept}: {description}" if description else concept
            for concept, description in zip(concepts, descriptions)
        ]
    else:
        texts = concepts

    valid_pairs = [(concept, text) for concept, text in zip(concepts, texts) if concept and text]
    if not valid_pairs:
        raise ValueError("没有可编码的非空 concept/text。")

    valid_concepts, valid_texts = zip(*valid_pairs)
    return list(valid_concepts), list(valid_texts)


def build_prompted_texts(texts: Sequence[str], prompt_mode: str) -> List[List[str]]:
    if prompt_mode == "none":
        return [list(texts)]
    if prompt_mode == "pathology_templates":
        return [
            [template.format(text=text) for text in texts]
            for template in PATHOLOGY_PROMPT_TEMPLATES
        ]
    raise ValueError(f"不支持的 prompt_mode: {prompt_mode}")


def load_titan(model_name: str, device: torch.device):
    print(f"[INFO] Loading TITAN model: {model_name}")
    titan = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    titan = titan.to(device)
    titan.eval()

    text_encoder = titan.text_encoder.to(device)
    text_encoder.eval()
    tokenizer = titan.text_encoder.tokenizer
    print("[INFO] TITAN model loaded.")
    return titan, tokenizer


def encode_texts(
    titan,
    tokenizer,
    texts: Sequence[str],
    *,
    device: torch.device,
    batch_size: int,
    normalize: bool,
    prompt_mode: str,
) -> torch.Tensor:
    prompted_text_groups = build_prompted_texts(texts, prompt_mode)
    ensembled_embeddings: torch.Tensor | None = None

    for template_idx, prompted_texts in enumerate(prompted_text_groups, start=1):
        all_embeddings: List[torch.Tensor] = []
        desc = f"Template {template_idx}/{len(prompted_text_groups)}"
        if prompt_mode == "none":
            desc = "Encoding"
        else:
            print(f"[INFO] {desc}: {prompted_texts[0]}")

        for start in tqdm(range(0, len(prompted_texts), batch_size), desc=desc, unit="batch"):
            batch_texts = list(prompted_texts[start : start + batch_size])
            tokenized_batch = tokenizer(batch_texts).to(device)

            with torch.no_grad():
                batch_embeddings = titan.encode_text(tokenized_batch)

            all_embeddings.append(batch_embeddings.detach().float().cpu())

        template_embeddings = torch.cat(all_embeddings, dim=0)
        if ensembled_embeddings is None:
            ensembled_embeddings = torch.zeros_like(template_embeddings)
        ensembled_embeddings += template_embeddings

    if ensembled_embeddings is None:
        raise ValueError("没有生成任何 embedding。")

    final_embeddings = ensembled_embeddings / len(prompted_text_groups)
    if normalize:
        final_embeddings = F.normalize(final_embeddings, p=2, dim=1)
    return final_embeddings


def save_outputs(
    *,
    output_dir: Path,
    input_path: Path,
    dataset: str,
    concepts: Sequence[str],
    texts: Sequence[str],
    embeddings: torch.Tensor,
    args: argparse.Namespace,
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)

    text_suffix = "concept_desc" if args.include_patch_description else "concept"
    prompt_suffix = "pathology_prompts" if args.prompt_mode == "pathology_templates" else "plain"
    output_prefix = f"{dataset}_titan_{text_suffix}_{prompt_suffix}"

    embedding_path = output_dir / f"{output_prefix}_embeddings.pt"
    metadata_path = output_dir / f"{output_prefix}_metadata.csv"
    summary_path = output_dir / f"{output_prefix}_summary.json"

    torch.save(embeddings, embedding_path)
    metadata_df = pd.DataFrame(
        {
            "dataset": dataset,
            "row_idx": list(range(len(concepts))),
            "concept": list(concepts),
            "encoded_text": list(texts),
        }
    )
    metadata_df.to_csv(metadata_path, index=False, encoding="utf-8-sig")

    summary: Dict[str, object] = {
        "dataset": dataset,
        "input_csv": str(input_path),
        "embedding_path": str(embedding_path),
        "metadata_path": str(metadata_path),
        "model_name": args.model_name,
        "include_patch_description": args.include_patch_description,
        "description_column": args.description_column,
        "prompt_mode": args.prompt_mode,
        "prompt_templates": PATHOLOGY_PROMPT_TEMPLATES if args.prompt_mode == "pathology_templates" else [],
        "concept_count": len(concepts),
        "embedding_shape": list(embeddings.shape),
        "normalized": not args.no_normalize,
        "batch_size": args.batch_size,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


def main() -> None:
    args = parse_args()
    if args.all_tasks:
        datasets = DEFAULT_DATASETS
    elif args.task_name:
        datasets = [args.task_name]
    else:
        raise ValueError("请指定 --task_name，或使用 --all_tasks 显式处理四个数据集。")

    input_root = Path(args.input_root)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    normalize = not args.no_normalize

    print(f"[INFO] Device: {device}")
    print(f"[INFO] Datasets: {', '.join(datasets)}")
    print(
        "[INFO] "
        f"include_patch_description={args.include_patch_description}, "
        f"prompt_mode={args.prompt_mode}, normalize={normalize}"
    )

    work_items = []
    text_suffix = "concept_desc" if args.include_patch_description else "concept"
    prompt_suffix = "pathology_prompts" if args.prompt_mode == "pathology_templates" else "plain"
    for dataset in datasets:
        input_path = resolve_input_path(input_root, dataset, args.input_name)
        output_dir = input_path.parent / args.output_dir_name
        output_prefix = f"{dataset}_titan_{text_suffix}_{prompt_suffix}"
        embedding_path = output_dir / f"{output_prefix}_embeddings.pt"
        metadata_path = output_dir / f"{output_prefix}_metadata.csv"
        if embedding_path.is_file() and metadata_path.is_file() and not args.overwrite:
            print(f"[SKIP] {dataset}: 输出已存在，使用 --overwrite 可重新提取。")
            continue
        work_items.append((dataset, input_path, output_dir))

    if not work_items:
        print("[INFO] 没有需要处理的数据集。")
        return

    titan, tokenizer = load_titan(args.model_name, device)
    all_summaries = []

    for dataset, input_path, output_dir in work_items:
        print(f"\n========== {dataset} ==========")
        print(f"[INFO] Input: {input_path}")
        df = pd.read_csv(input_path, encoding="utf-8")
        concepts, texts = build_texts(
            df,
            concept_column=args.concept_column,
            description_column=args.description_column,
            include_patch_description=args.include_patch_description,
        )
        print(f"[INFO] Encoding {len(texts)} texts ...")
        embeddings = encode_texts(
            titan,
            tokenizer,
            texts,
            device=device,
            batch_size=args.batch_size,
            normalize=normalize,
            prompt_mode=args.prompt_mode,
        )
        summary = save_outputs(
            output_dir=output_dir,
            input_path=input_path,
            dataset=dataset,
            concepts=concepts,
            texts=texts,
            embeddings=embeddings,
            args=args,
        )
        all_summaries.append(summary)
        print(f"[DONE] {dataset}: embeddings shape = {tuple(embeddings.shape)}")
        print(f"[INFO] Saved to: {output_dir}")

    if args.all_tasks:
        combined_summary_path = input_root / "titan_feature_summary.json"
        with open(combined_summary_path, "w", encoding="utf-8") as f:
            json.dump(all_summaries, f, ensure_ascii=False, indent=2)
        print(f"\n[INFO] Combined summary saved to: {combined_summary_path}")


if __name__ == "__main__":
    main()
