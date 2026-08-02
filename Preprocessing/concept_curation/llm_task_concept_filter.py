"""
llm_task_concept_filter.py

对 UMLS 筛选后的病理 concept 做 LLM-assisted task relevance filtering。
使用 multi-LLM consensus / adjudication 机制得到最终 task-specific concept pool。

Pipeline:
  UMLS-filtered concepts
  → LLM-A 独立判断
  → LLM-B 独立判断
  → Consensus 融合
  → (可选) LLM-C adjudication 裁决 uncertain 样本
  → 输出 final task-specific concept pool


# CAMELYON16，不使用 adjudicator
python llm_task_concept_filter.py \
  --input_csv /home/junjianli/DyKoJ/umls_concepts/concepts_umls_cui_dedup_semantic_filtered.csv \
  --output_dir /home/junjianli/DyKoJ/umls_concepts/llm_filtered/CAMELYON16 \
  --task_name CAMELYON16 \
  --api_base "https://api.aipaibox.com/v1" \
  --api_key "YOUR_API_KEY" \
  --batch_size 50 \
  --score_merge_strategy mean \
  --resume
# 带 adjudicator
python llm_task_concept_filter.py \
  ... \
  --use_adjudicator \
  --llm_c_model "claude-opus-4-6-thinking"
# 调试模式（只处理前 200 个）
python llm_task_concept_filter.py \
  ... \
  --limit 200 \
  --batch_size 20


output_dir/
├── raw/
│   ├── llm_a_batch_0.txt      # LLM-A 原始响应
│   ├── llm_b_batch_0.txt      # LLM-B 原始响应
│   └── llm_c_batch_0.txt      # LLM-C 原始响应（若使用）
├── parsed/
│   ├── llm_a_batch_0.json     # LLM-A 解析结果
│   ├── llm_b_batch_0.json     # LLM-B 解析结果
│   └── llm_c_batch_0.json     # LLM-C 解析结果（若使用）
├── llm_a_results.csv
├── llm_b_results.csv
├── consensus_all.csv
├── consensus_keep.csv
├── consensus_drop.csv
├── consensus_uncertain.csv
├── final_task_concept_pool.csv    # 最终 concept pool
├── final_concept_list.txt         # 供 TITAN encoder 使用
├── summary.json
├── error_log.jsonl
├── progress.json
├── llm_c_adjudication_results.csv （若使用 adjudicator）
└── consensus_after_adjudication.csv（若使用 adjudicator）


python llm_task_concept_filter.py \
  --input_csv /home/junjianli/DyKoJ/umls_concepts/concepts_umls_cui_dedup_semantic_filtered.csv \
  --output_dir /home/junjianli/DyKoJ/umls_concepts/llm_filtered/CAMELYON16 \
  --task_name CAMELYON16 \
  --api_base "https://api.aipaibox.com/v1" \
  --api_key "YOUR_SHARED_OR_LLM_B_KEY" \
  --llm_a_api_base "https://api.deepseek.com/v1" \
  --llm_a_api_key "YOUR_DEEPSEEK_KEY" \
  --llm_a_model "deepseek-chat" \
  --llm_b_model "gpt-5.5" \
  --use_adjudicator \
  --llm_c_model "claude-opus-4-6-thinking" \
  --batch_size 50 \
  --score_merge_strategy mean \
  --resume


batch_size过大，容易导致截断，返回不了完整的结果。
python llm_task_concept_filter.py   --input_csv ./concepts_umls_cui_dedup_semantic_filtered.csv   --output_dir ./llm_filtered/TCGA-RCC   --task_name TCGA-RCC   --use_adjudicator   --batch_size 50   --score_merge_strategy mean   --resume
"""

import argparse
import ast
import json
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from tqdm import tqdm


# ===========================================================================
# 常量定义
# ===========================================================================

VALID_DECISIONS = {"keep", "drop", "uncertain"}

VALID_CATEGORIES = {
    "diagnostic_disease",
    "morphological_finding",
    "cellular_component",
    "tissue_context",
    "anatomical_context",
    "microenvironment",
    "non_task_organ",
    "non_visual_molecular",
    "procedure_or_treatment",
    "uncertain",
}

# 内置任务预设
TASK_PRESETS: Dict[str, Dict[str, Any]] = {
    "CAMELYON16": {
        "task_description": (
            "Breast cancer lymph node metastasis classification in H&E whole-slide images. "
            "The task is to distinguish normal lymph node tissue from lymph nodes containing "
            "metastatic breast carcinoma."
        ),
        "class_labels": ["Normal", "Tumor"],
    },
    "NSCLC": {
        "task_description": (
            "Non-small cell lung cancer subtype classification in H&E whole-slide images. "
            "The task is to distinguish lung adenocarcinoma from lung squamous cell carcinoma."
        ),
        "class_labels": ["LUAD", "LUSC"],
    },
    "TCGA-RCC": {
        "task_description": (
            "Renal cell carcinoma subtype classification in H&E whole-slide images. "
            "The task is to distinguish clear-cell renal cell carcinoma, papillary renal cell "
            "carcinoma, and chromophobe renal cell carcinoma."
        ),
        "class_labels": ["ccRCC", "pRCC", "chRCC"],
    },
    "UBC-OCEAN": {
        "task_description": (
            "Ovarian carcinoma histological subtype classification in H&E whole-slide images. "
            "The task is to distinguish major ovarian carcinoma subtypes in H&E whole-slide images."
        ),
        "class_labels": [
            "Clear Cell Ovarian Carcinoma", 
            "Endometrioid Carcinoma", 
            "High-Grade Serous Carcinoma", 
            "Low-Grade Serous Carcinoma", 
            "Mucinous Carcinoma"
    ],
    },
}




# ===========================================================================
# 命令行参数
# ===========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LLM-assisted task relevance filtering for UMLS-normalized pathology concepts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 必需参数
    # parser.add_argument("--input_csv", default=fr'F:\AA-Research\00000-Project\李君健-科研-VLM-分型预后\AAAA-Journal\DyKoJ\concepts_umls_cui_dedup_semantic_filtered.csv', help="输入 UMLS concept CSV 路径")
    parser.add_argument("--input_csv", default=fr'/home/junjianli/DyKoJ/umls_concepts/concepts_umls_cui_dedup_semantic_filtered.csv', help="输入 UMLS concept CSV 路径")
    parser.add_argument("--output_dir", default='./llm_filter_concept', help="输出目录")
    parser.add_argument(
        "--task_name",
        default='CAMELYON16',
        help="任务名称，内置预设: CAMELYON16, NSCLC, TCGA-RCC, UBC-OCEAN。也可自定义。",
    )

    # 可选任务描述（覆盖预设）
    parser.add_argument("--task_description", default=None, help="任务描述，覆盖内置预设")
    parser.add_argument(
        "--class_labels",
        default=None,
        help="类别标签，逗号分隔，例如 'Normal,Tumor'，覆盖内置预设",
    )

    # 批处理参数
    parser.add_argument("--batch_size", type=int, default=50, help="每个 batch 的 concept 数量")
    parser.add_argument(
        "--batch_workers",
        type=int,
        default=20,
        help=(
            "并行处理 batch 的进程数（1=串行）。作用于 LLM-A、LLM-B（filter）"
            "及可选的 LLM-C（adjudicator）；各阶段独立建进程池。"
            ">1 易触发 API 限速，请酌情调小"
        ),
    )
    parser.add_argument("--max_retries", type=int, default=3, help="API 调用最大重试次数")
    parser.add_argument("--sleep_seconds", type=float, default=1.0, help="重试间隔秒数")

    # 运行控制
    parser.add_argument("--resume", action="store_true", help="断点续跑，跳过已完成的 batch")
    parser.add_argument(
        "--debug_llm_parse",
        action="store_true",
        help="解析与输入 concept 不一致时，在终端打印完整 LLM 原始文本（便于对照缺失/多余条目）",
    )
    parser.add_argument(
        "--debug_llm_parse_max_chars",
        type=int,
        default=80000,
        help="与 --debug_llm_parse 联用：原始文本最大打印字符数；0 表示不截断",
    )
    parser.add_argument(
        "--score_merge_strategy",
        choices=["mean", "min"],
        default="mean",
        help="合并两个 LLM 分数的策略",
    )
    parser.add_argument(
        "--keep_uncertain",
        action="store_true",
        help="最终输出中保留 uncertain 样本",
    )
    parser.add_argument(
        "--use_adjudicator",
        action="store_true",
        help="使用 LLM-C 对 uncertain 样本进行裁决",
    )
    parser.add_argument("--limit", type=int, default=None, help="调试用：只处理前 N 个 concepts")
    parser.add_argument("--start_index", type=int, default=0, help="从第 N 个 concept 开始处理")

    # 通用 API 参数
    parser.add_argument("--api_base", default="https://api.aipaibox.com/v1", help="通用 API base URL")
    parser.add_argument("--api_key", default=None, help="通用 API key")

    # LLM-A 参数
    parser.add_argument("--llm_a_api_base", default="https://us.novaiapi.com/v1", help="LLM-A API base URL（覆盖通用，留空则使用 --api_base）")
    parser.add_argument("--llm_a_api_key", default='sk-11111111111111111111111111111111111111', help="LLM-A API key（覆盖通用，留空则使用 --api_key）")
    parser.add_argument("--llm_a_model", default="[限时]claude-opus-4-6", help="LLM-A 模型名")

    parser.add_argument(
        "--llm_b_api_base",
        default="https://api.deepseek.com/v1",
        help="LLM-A API base URL（覆盖通用，留空则使用 --api_base）",
    )
    parser.add_argument(
        "--llm_b_api_key",
        default='sk-11111111111111111111111111111111111111',
        help="LLM-A API key（覆盖通用，留空则使用 --api_key）",
    )
    parser.add_argument("--llm_b_model", default="deepseek-v4-pro", help="LLM-A 模型名")

    # LLM-C adjudicator 参数
    parser.add_argument("--llm_c_api_base", default="https://us.novaiapi.com/v1", help="LLM-C API base URL（覆盖通用，留空则使用 --api_base）")
    parser.add_argument("--llm_c_api_key", default='sk-11111111111111111111111111111111111111', help="LLM-A API key（覆盖通用，留空则使用 --api_key）")
    parser.add_argument(
        "--llm_c_model", default="[次]claude-opus-4-7-thinking", help="LLM-C adjudicator 模型名"
    )

    return parser.parse_args()


# ===========================================================================
# API 配置解析
# ===========================================================================

def resolve_api_config(
    specific_base: Optional[str],
    specific_key: Optional[str],
    general_base: Optional[str],
    general_key: Optional[str],
) -> Tuple[str, str]:
    """解析单个 LLM 的 api_base 和 api_key，优先使用单独配置，否则 fallback 到通用配置。"""
    api_base = specific_base or general_base
    api_key = specific_key or general_key

    if not api_base:
        raise ValueError(
            "未找到有效的 api_base。请通过 --api_base 或 --llm_x_api_base 指定。"
        )
    if not api_key:
        raise ValueError(
            "未找到有效的 api_key。请通过 --api_key 或 --llm_x_api_key 指定。"
        )

    return api_base, api_key


def normalize_api_endpoint(api_base: str) -> str:
    """将 api_base 标准化为以 /chat/completions 结尾的完整 endpoint。"""
    if not api_base or not (api_base.startswith("http://") or api_base.startswith("https://")):
        raise ValueError(
            f"api_base 不是有效的 URL: '{api_base}'。\n"
            "请确保 --api_base 或 --llm_x_api_base 传入的是完整 URL，例如：\n"
            "  https://api.aipaibox.com/v1\n"
            "  https://api.deepseek.com/v1\n"
            "注意：--llm_x_api_base 是 API 地址，--llm_x_model 才是模型名称，请勿混淆。"
        )
    base = api_base.rstrip("/")
    # 如果已经以 /chat/completions 结尾，直接返回
    if base.endswith("/chat/completions"):
        return base
    # 如果以 /v1 结尾，拼接
    if base.endswith("/v1"):
        return base + "/chat/completions"
    # 其他情况，假设是 base URL，尝试拼接
    return base + "/chat/completions"


# ===========================================================================
# OpenAI-compatible API 调用
# ===========================================================================

def call_chat_completion(
    api_base: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.0,
    max_retries: int = 3,
    sleep_seconds: float = 1.0,
) -> str:
    """
    调用 OpenAI-compatible chat completion API，返回 assistant 消息内容字符串。
    使用 requests 实现，不依赖 OpenAI Python SDK。
    支持重试机制。
    """
    endpoint = normalize_api_endpoint(api_base)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }

    # 打印 key 的掩码版本用于 debug
    masked_key = api_key[:8] + "***" if len(api_key) > 8 else "***"

    last_error: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=120,
            )
            if response.status_code != 200:
                err_text = response.text[:300]
                print(
                    f"  [API 错误] 状态码 {response.status_code}，尝试 {attempt}/{max_retries}。"
                    f" Endpoint: {endpoint}，Key: {masked_key}。响应: {err_text}"
                )
                last_error = RuntimeError(f"HTTP {response.status_code}: {err_text}")
                time.sleep(sleep_seconds * attempt)
                continue

            data = response.json()
            content = data["choices"][0]["message"]["content"]
            return content

        except requests.exceptions.RequestException as e:
            print(f"  [网络错误] 尝试 {attempt}/{max_retries}: {e}")
            last_error = e
            time.sleep(sleep_seconds * attempt)
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            print(f"  [响应解析错误] 尝试 {attempt}/{max_retries}: {e}")
            last_error = e
            time.sleep(sleep_seconds * attempt)

    raise RuntimeError(
        f"API 调用失败，已重试 {max_retries} 次。最后错误: {last_error}"
    )


# ===========================================================================
# Prompt 构造
# ===========================================================================

def build_filter_prompt(
    batch_df: pd.DataFrame,
    task_description: str,
    class_labels: List[str],
) -> List[Dict[str, str]]:
    """
    构造 LLM filter prompt。
    返回 [{"role": "system", "content": ...}, {"role": "user", "content": ...}]
    """
    system_msg = (
        "You are a board-certified computational pathologist assisting with concept "
        "filtering for H&E whole-slide image classification."
    )

    class_labels_str = json.dumps(class_labels)

    # 构造输入 concept 列表
    concept_list = []
    for _, row in batch_df.iterrows():
        item = {
            "concept": str(row.get("concept", "")),
            "cui": str(row.get("cui", "")),
            "umls_name": str(row.get("umls_name", "")),
            "umls_type_names": row.get("umls_type_names", ""),
            "freq": int(row.get("freq", 0)) if not pd.isna(row.get("freq", 0)) else 0,
            "umls_score": float(row.get("umls_score", 0.0)) if not pd.isna(row.get("umls_score", 0.0)) else 0.0,
        }
        # umls_type_names 可能是字符串形式的列表，尝试解析
        if isinstance(item["umls_type_names"], str):
            try:
                item["umls_type_names"] = ast.literal_eval(item["umls_type_names"])
            except Exception:
                item["umls_type_names"] = [item["umls_type_names"]]
        concept_list.append(item)

    num_concepts = len(concept_list)

    # 构造 class_relevance 示例
    class_rel_example = {label: 0.0 for label in class_labels}

    user_msg = f"""The input concepts have already been:
- UMLS-normalized
- CUI-deduplicated
- semantic-type-filtered

Your task is to evaluate whether each concept is relevant to the target WSI classification task.

Target task:
{task_description}

Class labels:
{class_labels_str}

CRITICAL — output integrity (required for automated parsing; violations break downstream code):
- Output a single JSON array containing exactly {num_concepts} objects — one object per input concept, same count as the "Input concepts" list below.
- Every object MUST include "concept" copied verbatim from the corresponding input object's "concept" string (identical characters: no paraphrasing, abbreviating, trimming, fixing typos, or changing capitalization).
- Preserve one-to-one coverage: do not skip any input concept, do not add concepts not in the input, and do not merge multiple inputs into one row.
- Use the same conceptual order as the input list (first output row ↔ first input concept, and so on) to reduce mistakes.

Evaluation criteria:
1. Keep concepts related to the target disease, histological subtype, H&E-visible tumor morphology, cellular morphology, tissue architecture, tumor microenvironment, inflammation, necrosis, invasion, or task-relevant anatomy.
2. Drop concepts from unrelated organs, unrelated diseases, unrelated tumor types, molecular markers, genes, proteins, drugs, treatments, lab procedures, devices, or non-H&E-visible entities.
3. Keep anatomical concepts only if they are relevant to the target organ or diagnostic setting.
4. Be conservative. If a concept is broadly medical but not clearly useful for the target task, mark it as uncertain.
5. Do not introduce new concepts.
6. Return only valid JSON.

Allowed decisions:
["keep", "drop", "uncertain"]

Allowed categories:
[
  "diagnostic_disease",
  "morphological_finding",
  "cellular_component",
  "tissue_context",
  "anatomical_context",
  "microenvironment",
  "non_task_organ",
  "non_visual_molecular",
  "procedure_or_treatment",
  "uncertain"
]

Score meaning:
0.90-1.00: strongly task-relevant, direct diagnostic concept or core morphology
0.70-0.89: moderately/highly relevant tissue, cell, microenvironment, or morphology concept
0.40-0.69: weakly relevant or optional background concept
0.00-0.39: irrelevant or should be removed

For each concept, return:
- concept (must match the input "concept" string exactly, character-for-character)
- decision
- category
- task_relevance (float 0.0-1.0)
- class_relevance (dict with keys {class_labels_str}, each float 0.0-1.0)
- reason (concise, ≤20 words)

Input concepts:
{json.dumps(concept_list, ensure_ascii=False, indent=2)}

Expected output format:
[
  {{
    "concept": "...",
    "decision": "keep",
    "category": "...",
    "task_relevance": 0.0,
    "class_relevance": {json.dumps(class_rel_example)},
    "reason": "..."
  }}
]

Return only JSON. Do not include markdown code fences. Do not include explanations outside JSON.
Remember: the array length must be {num_concepts} and every "concept" must match the input verbatim."""

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


def build_adjudication_prompt(
    disagreement_df: pd.DataFrame,
    task_description: str,
    class_labels: List[str],
    llm_a_results: Dict[str, Dict],
    llm_b_results: Dict[str, Dict],
) -> List[Dict[str, str]]:
    """
    构造 LLM-C adjudicator prompt。
    disagreement_df 是需要裁决的 concept 集合。
    llm_a_results / llm_b_results 是 {concept: parsed_item} 字典。
    """
    system_msg = (
        "You are a senior board-certified computational pathologist serving as an "
        "independent adjudicator for task-relevant concept selection."
    )

    class_labels_str = json.dumps(class_labels)
    class_rel_example = {label: 0.0 for label in class_labels}

    # 构造带有 LLM-A 和 LLM-B 结果的输入列表
    items = []
    for _, row in disagreement_df.iterrows():
        concept = str(row.get("concept", ""))
        a = llm_a_results.get(concept, {})
        b = llm_b_results.get(concept, {})

        umls_type_names = row.get("umls_type_names", "")
        if isinstance(umls_type_names, str):
            try:
                umls_type_names = ast.literal_eval(umls_type_names)
            except Exception:
                umls_type_names = [umls_type_names]

        item = {
            "concept": concept,
            "cui": str(row.get("cui", "")),
            "umls_name": str(row.get("umls_name", "")),
            "umls_type_names": umls_type_names,
            "freq": int(row.get("freq", 0)) if not pd.isna(row.get("freq", 0)) else 0,
            "umls_score": float(row.get("umls_score", 0.0)) if not pd.isna(row.get("umls_score", 0.0)) else 0.0,
            "llm_a": {
                "decision": a.get("decision", "uncertain"),
                "category": a.get("category", "uncertain"),
                "task_relevance": a.get("task_relevance", 0.0),
                "class_relevance": a.get("class_relevance", {}),
                "reason": a.get("reason", ""),
            },
            "llm_b": {
                "decision": b.get("decision", "uncertain"),
                "category": b.get("category", "uncertain"),
                "task_relevance": b.get("task_relevance", 0.0),
                "class_relevance": b.get("class_relevance", {}),
                "reason": b.get("reason", ""),
            },
        }
        items.append(item)

    num_adjudication_items = len(items)

    user_msg = f"""Target task:
{task_description}

Class labels:
{class_labels_str}

You are given UMLS-normalized pathology concepts and two independent LLM judgments. Your task is to make the final decision.

CRITICAL — output integrity (required for automated parsing):
- Return a single JSON array with exactly {num_adjudication_items} objects — one per row in "Input concepts with LLM-A and LLM-B judgments" below.
- Each object's "concept" MUST copy the input "concept" string verbatim (same characters; no renaming or paraphrasing).
- Do not skip, add, or merge rows; keep output order aligned with the input list order.

Adjudication criteria:
1. Keep a concept only if it is useful for H&E-visible morphology, target disease diagnosis, target histological subtype, tissue architecture, cellular morphology, tumor microenvironment, inflammation, necrosis, invasion, or task-relevant anatomy.
2. Drop concepts from unrelated organs, unrelated diseases, unrelated tumor types, molecular markers, genes, proteins, drugs, procedures, lab tests, devices, or non-H&E-visible entities.
3. Anatomical context should be kept only if relevant to the target organ or diagnostic setting.
4. If evidence is insufficient, choose uncertain.
5. Do not introduce new concepts.
6. Return only valid JSON.

Input concepts with LLM-A and LLM-B judgments:
{json.dumps(items, ensure_ascii=False, indent=2)}

Output schema (same as input evaluations):
[
  {{
    "concept": "...",
    "decision": "keep",
    "category": "...",
    "task_relevance": 0.0,
    "class_relevance": {json.dumps(class_rel_example)},
    "reason": "..."
  }}
]

Return only JSON. Do not include markdown code fences.
Remember: {num_adjudication_items} objects, each "concept" verbatim from the input."""

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


# ===========================================================================
# JSON 解析与校验
# ===========================================================================

def _extract_json_array(text: str) -> Optional[str]:
    """从文本中提取第一个有效 JSON array 字符串。"""
    # 先尝试去除 markdown code fence
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if fence_match:
        return fence_match.group(1).strip()

    # 尝试找到第一个 '[' 到最后一个 ']'
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]

    return None


def parse_llm_json_response(
    text: str,
    expected_concepts: List[str],
    class_labels: List[str],
    error_log_path: Optional[Path] = None,
    *,
    llm_name: str = "",
    batch_id: Optional[int] = None,
    raw_path: Optional[Path] = None,
    debug_parse: bool = False,
    debug_max_chars: int = 80000,
    error_log_buffer: Optional[List[Dict[str, Any]]] = None,
    warning_lines: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    解析 LLM 返回的 JSON 字符串，并做字段校验与容错处理。
    返回标准化后的 list，每个 item 对应一个 concept。
    """
    parsed: Optional[List] = None

    # 尝试直接解析
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        pass

    # 尝试从 markdown fence 或原始文本中提取 JSON
    if parsed is None:
        extracted = _extract_json_array(text)
        if extracted:
            try:
                parsed = json.loads(extracted)
            except json.JSONDecodeError as e:
                _log_error(
                    error_log_path,
                    {"error": "JSON 解析失败", "detail": str(e), "raw_snippet": text[:200]},
                    error_log_buffer=error_log_buffer,
                )

    if parsed is None or not isinstance(parsed, list):
        _log_error(
            error_log_path,
            {"error": "LLM 返回值不是有效 JSON list", "raw_snippet": text[:200]},
            error_log_buffer=error_log_buffer,
        )
        parsed = []

    # 建立 concept -> item 映射，方便查找
    concept_to_item: Dict[str, Dict] = {}
    unexpected_from_llm: List[str] = []
    for item in parsed:
        if not isinstance(item, dict):
            _log_error(
                error_log_path,
                {"error": "LLM 返回 item 不是 dict", "item": str(item)[:100]},
                error_log_buffer=error_log_buffer,
            )
            continue

        concept_raw = item.get("concept", "")
        concept = str(concept_raw) if concept_raw is not None else ""
        if concept not in expected_concepts:
            # concept 不在输入列表中，记录警告但跳过
            unexpected_from_llm.append(concept)
            _log_error(
                error_log_path,
                {
                    "warning": "LLM 返回了不在输入列表中的 concept",
                    "concept": concept,
                },
                error_log_buffer=error_log_buffer,
            )
            continue

        # 校验并规范化各字段
        normalized = _normalize_item(
            item, class_labels, error_log_path, error_log_buffer=error_log_buffer
        )
        concept_to_item[concept] = normalized

    missing_concepts = [c for c in expected_concepts if c not in concept_to_item]
    if missing_concepts or unexpected_from_llm:
        batch_label = batch_id if batch_id is not None else "?"
        raw_hint = str(raw_path.resolve()) if raw_path else "(未传入路径，请检查 raw_dir 下对应 batch 的 .txt)"
        _log_error(
            error_log_path,
            {
                "warning": "LLM 解析与输入 concept 不一致（摘要）",
                "llm_name": llm_name,
                "batch_id": batch_id,
                "raw_path": raw_hint,
                "expected_count": len(expected_concepts),
                "matched_count": len(concept_to_item),
                "missing_count": len(missing_concepts),
                "missing_sample": missing_concepts[:30],
                "unexpected_count": len(unexpected_from_llm),
                "unexpected_sample": unexpected_from_llm[:30],
            },
            error_log_buffer=error_log_buffer,
        )
        summary_msg = (
            f"  [解析] {llm_name or 'LLM'} batch {batch_label}: "
            f"期望 {len(expected_concepts)} 条，匹配 {len(concept_to_item)} 条，"
            f"缺失 {len(missing_concepts)} 条，模型返回多余/无法对齐 {len(unexpected_from_llm)} 条。\n"
            f"           原始响应已写入: {raw_hint}"
        )
        if warning_lines is not None:
            warning_lines.append(summary_msg)
        else:
            print(summary_msg)
        if debug_parse:
            limit = len(text) if debug_max_chars <= 0 else debug_max_chars
            body = text if len(text) <= limit else text[:limit]
            truncated_note = (
                "" if len(text) <= limit else f"\n... （共 {len(text)} 字符，已按 --debug_llm_parse_max_chars 截断）"
            )
            dbg = (
                f"  --- LLM 原始输出 ({llm_name or 'LLM'} batch {batch_label}) ---\n"
                f"{body}{truncated_note}\n  --- 结束 ---"
            )
            if warning_lines is not None:
                warning_lines.append(dbg)
            else:
                print(dbg)

    # 对未返回的 concept 创建 fallback item
    result = []
    for concept in expected_concepts:
        if concept in concept_to_item:
            result.append(concept_to_item[concept])
        else:
            _log_error(
                error_log_path,
                {"warning": "concept 未出现在 LLM 返回中，使用 fallback", "concept": concept},
                error_log_buffer=error_log_buffer,
            )
            fallback = {
                "concept": concept,
                "decision": "uncertain",
                "category": "uncertain",
                "task_relevance": 0.0,
                "class_relevance": {label: 0.0 for label in class_labels},
                "reason": "Missing from LLM output.",
            }
            result.append(fallback)

    return result


def _normalize_item(
    item: Dict,
    class_labels: List[str],
    error_log_path: Optional[Path] = None,
    error_log_buffer: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """对单个 LLM 输出 item 做字段规范化和容错。"""
    concept = str(item.get("concept", ""))

    # decision 校验
    decision = str(item.get("decision", "uncertain")).lower().strip()
    if decision not in VALID_DECISIONS:
        _log_error(
            error_log_path,
            {"warning": f"非法 decision '{decision}'，改为 uncertain", "concept": concept},
            error_log_buffer=error_log_buffer,
        )
        decision = "uncertain"

    # category 校验
    category = str(item.get("category", "uncertain")).lower().strip()
    if category not in VALID_CATEGORIES:
        _log_error(
            error_log_path,
            {"warning": f"非法 category '{category}'，改为 uncertain", "concept": concept},
            error_log_buffer=error_log_buffer,
        )
        category = "uncertain"

    # task_relevance
    try:
        task_relevance = float(item.get("task_relevance", 0.0))
        task_relevance = max(0.0, min(1.0, task_relevance))
    except (TypeError, ValueError):
        task_relevance = 0.0

    # class_relevance
    raw_cr = item.get("class_relevance", {})
    if not isinstance(raw_cr, dict):
        raw_cr = {}

    class_relevance: Dict[str, float] = {}
    for label in class_labels:
        try:
            score = float(raw_cr.get(label, 0.0))
            score = max(0.0, min(1.0, score))
        except (TypeError, ValueError):
            score = 0.0
        class_relevance[label] = score

    reason = str(item.get("reason", "")).strip()

    return {
        "concept": concept,
        "decision": decision,
        "category": category,
        "task_relevance": task_relevance,
        "class_relevance": class_relevance,
        "reason": reason,
    }


def _log_error(
    error_log_path: Optional[Path],
    entry: Dict,
    *,
    error_log_buffer: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """追加写入错误日志；若提供 error_log_buffer（多进程 worker），则仅写入缓冲区由主进程统一落盘。"""
    line = dict(entry)
    line["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    if error_log_buffer is not None:
        error_log_buffer.append(line)
    if error_log_path is not None:
        try:
            with open(error_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
        except Exception:
            pass


# ===========================================================================
# Consensus 合并
# ===========================================================================

def _merge_score(a: float, b: float, strategy: str) -> float:
    if strategy == "mean":
        return (a + b) / 2.0
    elif strategy == "min":
        return min(a, b)
    return (a + b) / 2.0


def merge_llm_results(
    df_original: pd.DataFrame,
    results_a: List[Dict],
    results_b: List[Dict],
    class_labels: List[str],
    score_merge_strategy: str = "mean",
) -> pd.DataFrame:
    """
    合并 LLM-A 和 LLM-B 的结果，生成 consensus 结果。
    返回包含原始字段和 consensus 字段的 DataFrame。
    """
    # 建立 concept -> result 映射
    a_map = {item["concept"]: item for item in results_a}
    b_map = {item["concept"]: item for item in results_b}

    rows = []
    for _, orig_row in df_original.iterrows():
        concept = str(orig_row.get("concept", ""))

        a = a_map.get(concept, {
            "concept": concept, "decision": "uncertain", "category": "uncertain",
            "task_relevance": 0.0, "class_relevance": {l: 0.0 for l in class_labels},
            "reason": "Missing from LLM-A output.",
        })
        b = b_map.get(concept, {
            "concept": concept, "decision": "uncertain", "category": "uncertain",
            "task_relevance": 0.0, "class_relevance": {l: 0.0 for l in class_labels},
            "reason": "Missing from LLM-B output.",
        })

        dec_a = a["decision"]
        dec_b = b["decision"]

        # Consensus 决策规则
        if dec_a == "keep" and dec_b == "keep":
            final_decision = "keep"
            consensus_note = "Consensus from LLM-A and LLM-B."
        elif dec_a == "drop" and dec_b == "drop":
            final_decision = "drop"
            consensus_note = "Consensus from LLM-A and LLM-B."
        else:
            final_decision = "uncertain"
            consensus_note = "Disagreement between LLM-A and LLM-B."

        # task_relevance 合并
        final_task_relevance = _merge_score(
            a["task_relevance"], b["task_relevance"], score_merge_strategy
        )

        # class_relevance 合并
        final_class_relevance: Dict[str, float] = {}
        for label in class_labels:
            sa = a["class_relevance"].get(label, 0.0)
            sb = b["class_relevance"].get(label, 0.0)
            final_class_relevance[label] = _merge_score(sa, sb, score_merge_strategy)

        # category 合并
        if a["category"] == b["category"]:
            final_category = a["category"]
        else:
            final_category = "uncertain"

        row = dict(orig_row)
        row.update({
            "llm_a_decision": dec_a,
            "llm_a_category": a["category"],
            "llm_a_task_relevance": a["task_relevance"],
            "llm_a_class_relevance_json": json.dumps(a["class_relevance"], ensure_ascii=False),
            "llm_a_reason": a["reason"],
            "llm_b_decision": dec_b,
            "llm_b_category": b["category"],
            "llm_b_task_relevance": b["task_relevance"],
            "llm_b_class_relevance_json": json.dumps(b["class_relevance"], ensure_ascii=False),
            "llm_b_reason": b["reason"],
            "final_decision": final_decision,
            "final_category": final_category,
            "final_task_relevance": final_task_relevance,
            "final_class_relevance_json": json.dumps(final_class_relevance, ensure_ascii=False),
            "final_reason": consensus_note,
            "adjudicator_used": False,
            # LLM-C 字段占位
            "llm_c_decision": "",
            "llm_c_category": "",
            "llm_c_task_relevance": None,
            "llm_c_class_relevance_json": "",
            "llm_c_reason": "",
        })
        rows.append(row)

    return pd.DataFrame(rows)


# ===========================================================================
# 进度管理
# ===========================================================================

def load_progress(progress_path: Path) -> Dict:
    if progress_path.exists():
        try:
            with open(progress_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_progress(progress_path: Path, progress: Dict) -> None:
    with open(progress_path, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


def _flush_error_buffer(error_log_path: Path, buffer: List[Dict[str, Any]]) -> None:
    """将子进程收集的日志条目追加写入 error_log.jsonl（主进程调用）。"""
    if not buffer:
        return
    try:
        with open(error_log_path, "a", encoding="utf-8") as f:
            for line in buffer:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _mp_run_filter_batch(payload: Dict[str, Any]) -> Dict[str, Any]:
    """子进程：处理单个 filter batch（须为顶层函数以便 pickle）。"""
    batch_id = payload["batch_id"]
    batch_df = pd.DataFrame.from_records(payload["batch_records"])
    batch_concepts: List[str] = payload["batch_concepts"]
    task_description: str = payload["task_description"]
    class_labels: List[str] = payload["class_labels"]
    api_base: str = payload["api_base"]
    api_key: str = payload["api_key"]
    model: str = payload["model"]
    llm_name: str = payload["llm_name"]
    raw_path = Path(payload["raw_path"])
    parsed_path = Path(payload["parsed_path"])
    max_retries: int = payload["max_retries"]
    sleep_seconds: float = payload["sleep_seconds"]
    resume: bool = payload["resume"]
    debug_llm_parse: bool = payload["debug_llm_parse"]
    debug_llm_parse_max_chars: int = payload["debug_llm_parse_max_chars"]

    error_buf: List[Dict[str, Any]] = []
    warn_lines: List[str] = []

    if resume and parsed_path.exists():
        try:
            with open(parsed_path, "r", encoding="utf-8") as f:
                batch_results = json.load(f)
            return {
                "batch_id": batch_id,
                "results": batch_results,
                "error_log_buffer": error_buf,
                "warning_lines": warn_lines,
            }
        except Exception:
            pass

    messages = build_filter_prompt(batch_df, task_description, class_labels)

    try:
        raw_text = call_chat_completion(
            api_base=api_base,
            api_key=api_key,
            model=model,
            messages=messages,
            temperature=0.0,
            max_retries=max_retries,
            sleep_seconds=sleep_seconds,
        )
    except Exception as e:
        print(f"  [错误] {llm_name} batch {batch_id} API 调用失败: {e}")
        _log_error(
            None,
            {
                "error": f"{llm_name} batch {batch_id} API 调用失败",
                "detail": str(e),
                "batch_id": batch_id,
                "llm": llm_name,
            },
            error_log_buffer=error_buf,
        )
        fallback_results = [
            {
                "concept": c,
                "decision": "uncertain",
                "category": "uncertain",
                "task_relevance": 0.0,
                "class_relevance": {label: 0.0 for label in class_labels},
                "reason": f"API call failed: {str(e)[:50]}",
            }
            for c in batch_concepts
        ]
        raw_path.write_text(f"API call failed: {e}", encoding="utf-8")
        with open(parsed_path, "w", encoding="utf-8") as f:
            json.dump(fallback_results, f, ensure_ascii=False, indent=2)
        return {
            "batch_id": batch_id,
            "results": fallback_results,
            "error_log_buffer": error_buf,
            "warning_lines": warn_lines,
        }

    raw_path.write_text(raw_text, encoding="utf-8")
    batch_results = parse_llm_json_response(
        raw_text,
        batch_concepts,
        class_labels,
        None,
        llm_name=llm_name,
        batch_id=batch_id,
        raw_path=raw_path,
        debug_parse=debug_llm_parse,
        debug_max_chars=debug_llm_parse_max_chars,
        error_log_buffer=error_buf,
        warning_lines=warn_lines,
    )
    with open(parsed_path, "w", encoding="utf-8") as f:
        json.dump(batch_results, f, ensure_ascii=False, indent=2)

    return {
        "batch_id": batch_id,
        "results": batch_results,
        "error_log_buffer": error_buf,
        "warning_lines": warn_lines,
    }


def _mp_run_adjudication_batch(payload: Dict[str, Any]) -> Dict[str, Any]:
    """子进程：处理单个 LLM-C adjudication batch。"""
    batch_id = payload["batch_id"]
    batch_df = pd.DataFrame.from_records(payload["batch_records"])
    batch_concepts: List[str] = payload["batch_concepts"]
    task_description: str = payload["task_description"]
    class_labels: List[str] = payload["class_labels"]
    a_partial: Dict[str, Dict] = payload["a_partial"]
    b_partial: Dict[str, Dict] = payload["b_partial"]
    api_base: str = payload["api_base"]
    api_key: str = payload["api_key"]
    model: str = payload["model"]
    raw_path = Path(payload["raw_path"])
    parsed_path = Path(payload["parsed_path"])
    max_retries: int = payload["max_retries"]
    sleep_seconds: float = payload["sleep_seconds"]
    resume: bool = payload["resume"]
    debug_llm_parse: bool = payload["debug_llm_parse"]
    debug_llm_parse_max_chars: int = payload["debug_llm_parse_max_chars"]

    error_buf: List[Dict[str, Any]] = []
    warn_lines: List[str] = []

    if resume and parsed_path.exists():
        try:
            with open(parsed_path, "r", encoding="utf-8") as f:
                batch_results = json.load(f)
            return {
                "batch_id": batch_id,
                "results": batch_results,
                "error_log_buffer": error_buf,
                "warning_lines": warn_lines,
            }
        except Exception:
            pass

    messages = build_adjudication_prompt(
        batch_df, task_description, class_labels, a_partial, b_partial
    )

    try:
        raw_text = call_chat_completion(
            api_base=api_base,
            api_key=api_key,
            model=model,
            messages=messages,
            temperature=0.0,
            max_retries=max_retries,
            sleep_seconds=sleep_seconds,
        )
    except Exception as e:
        print(f"  [错误] LLM-C batch {batch_id} 调用失败: {e}")
        _log_error(
            None,
            {
                "error": f"LLM-C adjudicator batch {batch_id} 失败",
                "detail": str(e),
                "batch_id": batch_id,
            },
            error_log_buffer=error_buf,
        )
        fallback_results = [
            {
                "concept": c,
                "decision": "uncertain",
                "category": "uncertain",
                "task_relevance": 0.0,
                "class_relevance": {label: 0.0 for label in class_labels},
                "reason": f"LLM-C API call failed: {str(e)[:50]}",
            }
            for c in batch_concepts
        ]
        raw_path.write_text(f"API call failed: {e}", encoding="utf-8")
        with open(parsed_path, "w", encoding="utf-8") as f:
            json.dump(fallback_results, f, ensure_ascii=False, indent=2)
        return {
            "batch_id": batch_id,
            "results": fallback_results,
            "error_log_buffer": error_buf,
            "warning_lines": warn_lines,
        }

    raw_path.write_text(raw_text, encoding="utf-8")
    batch_results = parse_llm_json_response(
        raw_text,
        batch_concepts,
        class_labels,
        None,
        llm_name="llm_c",
        batch_id=batch_id,
        raw_path=raw_path,
        debug_parse=debug_llm_parse,
        debug_max_chars=debug_llm_parse_max_chars,
        error_log_buffer=error_buf,
        warning_lines=warn_lines,
    )
    with open(parsed_path, "w", encoding="utf-8") as f:
        json.dump(batch_results, f, ensure_ascii=False, indent=2)

    return {
        "batch_id": batch_id,
        "results": batch_results,
        "error_log_buffer": error_buf,
        "warning_lines": warn_lines,
    }


# ===========================================================================
# 单个 LLM 的 batch 处理
# ===========================================================================

def process_llm_for_all_batches(
    df: pd.DataFrame,
    batch_size: int,
    task_description: str,
    class_labels: List[str],
    api_base: str,
    api_key: str,
    model: str,
    llm_name: str,
    raw_dir: Path,
    parsed_dir: Path,
    error_log_path: Path,
    max_retries: int = 3,
    sleep_seconds: float = 1.0,
    resume: bool = True,
    debug_llm_parse: bool = False,
    debug_llm_parse_max_chars: int = 80000,
    batch_workers: int = 1,
) -> List[Dict]:
    """
    对所有 concepts 分 batch 调用单个 LLM，返回所有 batch 的 parsed results 合并列表。
    支持断点续跑；batch_workers>1 时多进程并行处理多个 batch。
    """
    concepts = df["concept"].tolist()
    num_batches = (len(df) + batch_size - 1) // batch_size

    if batch_workers > 1:
        results_by_batch: Dict[int, List[Dict]] = {}
        payloads: List[Dict[str, Any]] = []
        for batch_id in range(num_batches):
            batch_start = batch_id * batch_size
            batch_end = min(batch_start + batch_size, len(df))
            batch_df = df.iloc[batch_start:batch_end]
            batch_concepts = concepts[batch_start:batch_end]
            raw_path = raw_dir / f"{llm_name}_batch_{batch_id}.txt"
            parsed_path = parsed_dir / f"{llm_name}_batch_{batch_id}.json"

            if resume and parsed_path.exists():
                try:
                    with open(parsed_path, "r", encoding="utf-8") as f:
                        results_by_batch[batch_id] = json.load(f)
                    continue
                except Exception as e:
                    print(f"  [警告] 加载已有 parsed 失败，将重跑: {parsed_path} ({e})")

            payloads.append(
                {
                    "batch_id": batch_id,
                    "batch_records": batch_df.to_dict("records"),
                    "batch_concepts": batch_concepts,
                    "task_description": task_description,
                    "class_labels": class_labels,
                    "api_base": api_base,
                    "api_key": api_key,
                    "model": model,
                    "llm_name": llm_name,
                    "raw_path": str(raw_path),
                    "parsed_path": str(parsed_path),
                    "max_retries": max_retries,
                    "sleep_seconds": sleep_seconds,
                    "resume": False,
                    "debug_llm_parse": debug_llm_parse,
                    "debug_llm_parse_max_chars": debug_llm_parse_max_chars,
                }
            )

        if payloads:
            workers = min(batch_workers, len(payloads))
            print(
                f"  [并行] {llm_name}: {workers} 进程，共 {len(payloads)} 个待处理 batch "
                f"（已有 {len(results_by_batch)} 个由 resume 跳过）"
            )
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(_mp_run_filter_batch, p) for p in payloads]
                for fut in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"{llm_name} ({model}) 并行 batches",
                ):
                    out = fut.result()
                    bid = out["batch_id"]
                    results_by_batch[bid] = out["results"]
                    _flush_error_buffer(error_log_path, out["error_log_buffer"])
                    for line in out["warning_lines"]:
                        print(line)

        all_results: List[Dict] = []
        for batch_id in range(num_batches):
            all_results.extend(results_by_batch[batch_id])
        return all_results

    all_results = []

    for batch_id in tqdm(range(num_batches), desc=f"{llm_name} ({model}) 处理 batches"):
        batch_start = batch_id * batch_size
        batch_end = min(batch_start + batch_size, len(df))
        batch_df = df.iloc[batch_start:batch_end]
        batch_concepts = concepts[batch_start:batch_end]

        raw_path = raw_dir / f"{llm_name}_batch_{batch_id}.txt"
        parsed_path = parsed_dir / f"{llm_name}_batch_{batch_id}.json"

        # 断点续跑：如果 parsed 已存在，直接加载
        if resume and parsed_path.exists():
            try:
                with open(parsed_path, "r", encoding="utf-8") as f:
                    batch_results = json.load(f)
                all_results.extend(batch_results)
                continue
            except Exception as e:
                print(f"  [警告] 加载已有 parsed 文件失败，重新处理: {parsed_path} ({e})")

        # 构造 prompt
        messages = build_filter_prompt(batch_df, task_description, class_labels)

        # 调用 API
        try:
            raw_text = call_chat_completion(
                api_base=api_base,
                api_key=api_key,
                model=model,
                messages=messages,
                temperature=0.0,
                max_retries=max_retries,
                sleep_seconds=sleep_seconds,
            )
        except Exception as e:
            print(f"  [错误] {llm_name} batch {batch_id} API 调用失败: {e}")
            _log_error(
                error_log_path,
                {
                    "error": f"{llm_name} batch {batch_id} API 调用失败",
                    "detail": str(e),
                    "batch_id": batch_id,
                    "llm": llm_name,
                },
            )
            # 为该 batch 的所有 concept 生成 fallback
            fallback_results = [
                {
                    "concept": c,
                    "decision": "uncertain",
                    "category": "uncertain",
                    "task_relevance": 0.0,
                    "class_relevance": {label: 0.0 for label in class_labels},
                    "reason": f"API call failed: {str(e)[:50]}",
                }
                for c in batch_concepts
            ]
            all_results.extend(fallback_results)

            # 保存 fallback 到 raw 和 parsed
            raw_path.write_text(f"API call failed: {e}", encoding="utf-8")
            with open(parsed_path, "w", encoding="utf-8") as f:
                json.dump(fallback_results, f, ensure_ascii=False, indent=2)
            continue

        # 保存 raw output
        raw_path.write_text(raw_text, encoding="utf-8")

        # 解析 JSON
        batch_results = parse_llm_json_response(
            raw_text,
            batch_concepts,
            class_labels,
            error_log_path,
            llm_name=llm_name,
            batch_id=batch_id,
            raw_path=raw_path,
            debug_parse=debug_llm_parse,
            debug_max_chars=debug_llm_parse_max_chars,
        )

        # 保存 parsed output
        with open(parsed_path, "w", encoding="utf-8") as f:
            json.dump(batch_results, f, ensure_ascii=False, indent=2)

        all_results.extend(batch_results)

        # 适当 sleep，避免 API 限速
        time.sleep(0.2)

    return all_results


# ===========================================================================
# LLM-C Adjudication
# ===========================================================================

def run_adjudication(
    uncertain_df: pd.DataFrame,
    df_original: pd.DataFrame,
    results_a: List[Dict],
    results_b: List[Dict],
    class_labels: List[str],
    task_description: str,
    api_base: str,
    api_key: str,
    model: str,
    batch_size: int,
    raw_dir: Path,
    parsed_dir: Path,
    error_log_path: Path,
    max_retries: int = 3,
    sleep_seconds: float = 1.0,
    resume: bool = True,
    debug_llm_parse: bool = False,
    debug_llm_parse_max_chars: int = 80000,
    batch_workers: int = 1,
) -> List[Dict]:
    """
    对 uncertain 的 concepts 调用 LLM-C adjudicator。
    返回 LLM-C 的 parsed results 列表。
    """
    _ = df_original  # 保留签名兼容，逻辑仍基于 uncertain_df
    a_map = {item["concept"]: item for item in results_a}
    b_map = {item["concept"]: item for item in results_b}

    concepts = uncertain_df["concept"].tolist()
    num_batches = (len(uncertain_df) + batch_size - 1) // batch_size

    print(f"\n[LLM-C adjudicator] 开始裁决 {len(uncertain_df)} 个 uncertain concepts，共 {num_batches} 个 batch...")

    if batch_workers > 1:
        results_by_batch: Dict[int, List[Dict]] = {}
        payloads: List[Dict[str, Any]] = []
        for batch_id in range(num_batches):
            batch_start = batch_id * batch_size
            batch_end = min(batch_start + batch_size, len(uncertain_df))
            batch_df = uncertain_df.iloc[batch_start:batch_end]
            batch_concepts = concepts[batch_start:batch_end]

            raw_path = raw_dir / f"llm_c_batch_{batch_id}.txt"
            parsed_path = parsed_dir / f"llm_c_batch_{batch_id}.json"

            if resume and parsed_path.exists():
                try:
                    with open(parsed_path, "r", encoding="utf-8") as f:
                        results_by_batch[batch_id] = json.load(f)
                    continue
                except Exception as e:
                    print(f"  [警告] 加载 LLM-C parsed 失败，将重跑: {e}")

            a_partial = {c: dict(a_map.get(c, {})) for c in batch_concepts}
            b_partial = {c: dict(b_map.get(c, {})) for c in batch_concepts}
            payloads.append(
                {
                    "batch_id": batch_id,
                    "batch_records": batch_df.to_dict("records"),
                    "batch_concepts": batch_concepts,
                    "task_description": task_description,
                    "class_labels": class_labels,
                    "a_partial": a_partial,
                    "b_partial": b_partial,
                    "api_base": api_base,
                    "api_key": api_key,
                    "model": model,
                    "raw_path": str(raw_path),
                    "parsed_path": str(parsed_path),
                    "max_retries": max_retries,
                    "sleep_seconds": sleep_seconds,
                    "resume": False,
                    "debug_llm_parse": debug_llm_parse,
                    "debug_llm_parse_max_chars": debug_llm_parse_max_chars,
                }
            )

        if payloads:
            workers = min(batch_workers, len(payloads))
            print(
                f"  [并行] LLM-C: {workers} 进程，{len(payloads)} 个待处理 batch "
                f"（resume 跳过 {len(results_by_batch)} 个）"
            )
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(_mp_run_adjudication_batch, p) for p in payloads]
                for fut in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="LLM-C 并行 adjudicator",
                ):
                    out = fut.result()
                    results_by_batch[out["batch_id"]] = out["results"]
                    _flush_error_buffer(error_log_path, out["error_log_buffer"])
                    for line in out["warning_lines"]:
                        print(line)

        all_results_c: List[Dict] = []
        for batch_id in range(num_batches):
            all_results_c.extend(results_by_batch[batch_id])
        return all_results_c

    all_results: List[Dict] = []

    for batch_id in tqdm(range(num_batches), desc="LLM-C adjudicator"):
        batch_start = batch_id * batch_size
        batch_end = min(batch_start + batch_size, len(uncertain_df))
        batch_df = uncertain_df.iloc[batch_start:batch_end]
        batch_concepts = concepts[batch_start:batch_end]

        raw_path = raw_dir / f"llm_c_batch_{batch_id}.txt"
        parsed_path = parsed_dir / f"llm_c_batch_{batch_id}.json"

        if resume and parsed_path.exists():
            try:
                with open(parsed_path, "r", encoding="utf-8") as f:
                    batch_results = json.load(f)
                all_results.extend(batch_results)
                continue
            except Exception as e:
                print(f"  [警告] 加载 LLM-C parsed 失败，重新处理: {e}")

        messages = build_adjudication_prompt(
            batch_df, task_description, class_labels, a_map, b_map
        )

        try:
            raw_text = call_chat_completion(
                api_base=api_base,
                api_key=api_key,
                model=model,
                messages=messages,
                temperature=0.0,
                max_retries=max_retries,
                sleep_seconds=sleep_seconds,
            )
        except Exception as e:
            print(f"  [错误] LLM-C batch {batch_id} 调用失败: {e}")
            _log_error(
                error_log_path,
                {
                    "error": f"LLM-C adjudicator batch {batch_id} 失败",
                    "detail": str(e),
                    "batch_id": batch_id,
                },
            )
            fallback_results = [
                {
                    "concept": c,
                    "decision": "uncertain",
                    "category": "uncertain",
                    "task_relevance": 0.0,
                    "class_relevance": {label: 0.0 for label in class_labels},
                    "reason": f"LLM-C API call failed: {str(e)[:50]}",
                }
                for c in batch_concepts
            ]
            all_results.extend(fallback_results)
            raw_path.write_text(f"API call failed: {e}", encoding="utf-8")
            with open(parsed_path, "w", encoding="utf-8") as f:
                json.dump(fallback_results, f, ensure_ascii=False, indent=2)
            continue

        raw_path.write_text(raw_text, encoding="utf-8")

        batch_results = parse_llm_json_response(
            raw_text,
            batch_concepts,
            class_labels,
            error_log_path,
            llm_name="llm_c",
            batch_id=batch_id,
            raw_path=raw_path,
            debug_parse=debug_llm_parse,
            debug_max_chars=debug_llm_parse_max_chars,
        )

        with open(parsed_path, "w", encoding="utf-8") as f:
            json.dump(batch_results, f, ensure_ascii=False, indent=2)

        all_results.extend(batch_results)
        time.sleep(0.2)

    return all_results


# ===========================================================================
# 输出文件保存
# ===========================================================================

def save_outputs(
    consensus_df: pd.DataFrame,
    output_dir: Path,
    class_labels: List[str],
    keep_uncertain: bool,
    llm_c_results: Optional[List[Dict]],
    args: argparse.Namespace,
) -> None:
    """保存所有输出文件。"""

    # LLM-A 和 LLM-B 单独结果
    llm_a_cols = [
        "concept", "cui", "umls_name", "freq",
        "llm_a_decision", "llm_a_category",
        "llm_a_task_relevance", "llm_a_class_relevance_json", "llm_a_reason",
    ]
    llm_b_cols = [
        "concept", "cui", "umls_name", "freq",
        "llm_b_decision", "llm_b_category",
        "llm_b_task_relevance", "llm_b_class_relevance_json", "llm_b_reason",
    ]
    _safe_save_csv(consensus_df[[c for c in llm_a_cols if c in consensus_df.columns]],
                   output_dir / "llm_a_results.csv")
    _safe_save_csv(consensus_df[[c for c in llm_b_cols if c in consensus_df.columns]],
                   output_dir / "llm_b_results.csv")

    # Consensus 结果
    _safe_save_csv(consensus_df, output_dir / "consensus_all.csv")
    _safe_save_csv(
        consensus_df[consensus_df["final_decision"] == "keep"],
        output_dir / "consensus_keep.csv",
    )
    _safe_save_csv(
        consensus_df[consensus_df["final_decision"] == "drop"],
        output_dir / "consensus_drop.csv",
    )
    _safe_save_csv(
        consensus_df[consensus_df["final_decision"] == "uncertain"],
        output_dir / "consensus_uncertain.csv",
    )

    # 处理 LLM-C adjudication 结果
    final_df = consensus_df.copy()
    adjudicated_count = 0

    if llm_c_results:
        c_map = {item["concept"]: item for item in llm_c_results}

        # 将 LLM-C 结果更新到 final_df 中 uncertain 的行
        for idx, row in final_df.iterrows():
            if row["final_decision"] == "uncertain":
                concept = row["concept"]
                if concept in c_map:
                    c = c_map[concept]
                    final_df.at[idx, "llm_c_decision"] = c["decision"]
                    final_df.at[idx, "llm_c_category"] = c["category"]
                    final_df.at[idx, "llm_c_task_relevance"] = c["task_relevance"]
                    final_df.at[idx, "llm_c_class_relevance_json"] = json.dumps(
                        c["class_relevance"], ensure_ascii=False
                    )
                    final_df.at[idx, "llm_c_reason"] = c["reason"]
                    # 用 LLM-C 结果覆盖 final
                    final_df.at[idx, "final_decision"] = c["decision"]
                    final_df.at[idx, "final_category"] = c["category"]
                    final_df.at[idx, "final_task_relevance"] = c["task_relevance"]
                    final_df.at[idx, "final_class_relevance_json"] = json.dumps(
                        c["class_relevance"], ensure_ascii=False
                    )
                    final_df.at[idx, "final_reason"] = c["reason"]
                    final_df.at[idx, "adjudicator_used"] = True
                    adjudicated_count += 1

        # 保存 adjudication 相关文件
        llm_c_df = final_df[final_df["adjudicator_used"] == True]
        _safe_save_csv(llm_c_df, output_dir / "llm_c_adjudication_results.csv")
        _safe_save_csv(final_df, output_dir / "consensus_after_adjudication.csv")

    # final_task_concept_pool.csv
    if keep_uncertain:
        pool_df = final_df[final_df["final_decision"].isin(["keep", "uncertain"])]
    else:
        pool_df = final_df[final_df["final_decision"] == "keep"]

    _safe_save_csv(pool_df, output_dir / "final_task_concept_pool.csv")

    # final_concept_list.txt
    concept_list = pool_df["concept"].tolist()
    concept_list_path = output_dir / "final_concept_list.txt"
    with open(concept_list_path, "w", encoding="utf-8") as f:
        for c in concept_list:
            f.write(str(c) + "\n")

    # summary.json
    summary = {
        "task_name": args.task_name,
        "task_description": args.task_description,
        "class_labels": args.class_labels if isinstance(args.class_labels, list) else args.class_labels,
        "score_merge_strategy": args.score_merge_strategy,
        "keep_uncertain": args.keep_uncertain,
        "batch_size": args.batch_size,
        "batch_workers": args.batch_workers,
        "llm_a_model": args.llm_a_model,
        "llm_b_model": args.llm_b_model,
        "llm_c_model": args.llm_c_model,
        "input_concept_count": len(consensus_df),
        "processed_concept_count": len(consensus_df),
        "llm_a_keep_count": int((consensus_df["llm_a_decision"] == "keep").sum()),
        "llm_a_drop_count": int((consensus_df["llm_a_decision"] == "drop").sum()),
        "llm_a_uncertain_count": int((consensus_df["llm_a_decision"] == "uncertain").sum()),
        "llm_b_keep_count": int((consensus_df["llm_b_decision"] == "keep").sum()),
        "llm_b_drop_count": int((consensus_df["llm_b_decision"] == "drop").sum()),
        "llm_b_uncertain_count": int((consensus_df["llm_b_decision"] == "uncertain").sum()),
        "consensus_keep_count": int((consensus_df["final_decision"] == "keep").sum()),
        "consensus_drop_count": int((consensus_df["final_decision"] == "drop").sum()),
        "consensus_uncertain_count": int((consensus_df["final_decision"] == "uncertain").sum()),
        "adjudicator_used": args.use_adjudicator,
        "adjudicated_count": adjudicated_count,
        "final_keep_count": int((final_df["final_decision"] == "keep").sum()),
        "final_drop_count": int((final_df["final_decision"] == "drop").sum()),
        "final_uncertain_count": int((final_df["final_decision"] == "uncertain").sum()),
        "final_concept_count": len(pool_df),
    }

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 打印摘要
    print("\n" + "=" * 60)
    print(f"任务: {args.task_name}")
    print(f"输入 concepts: {summary['input_concept_count']}")
    print(f"LLM-A keep/drop/uncertain: {summary['llm_a_keep_count']} / {summary['llm_a_drop_count']} / {summary['llm_a_uncertain_count']}")
    print(f"LLM-B keep/drop/uncertain: {summary['llm_b_keep_count']} / {summary['llm_b_drop_count']} / {summary['llm_b_uncertain_count']}")
    print(f"Consensus keep/drop/uncertain: {summary['consensus_keep_count']} / {summary['consensus_drop_count']} / {summary['consensus_uncertain_count']}")
    if args.use_adjudicator:
        print(f"LLM-C 裁决数量: {adjudicated_count}")
    print(f"最终 concept pool 大小: {summary['final_concept_count']}")
    print(f"输出目录: {output_dir}")
    print("=" * 60)


def _safe_save_csv(df: pd.DataFrame, path: Path) -> None:
    """安全保存 CSV，忽略空 DataFrame 但仍创建文件。"""
    df.to_csv(path, index=False, encoding="utf-8-sig")


# ===========================================================================
# 主函数
# ===========================================================================

def main() -> None:
    args = parse_args()

    # 解析任务描述和 class_labels
    task_description: str
    class_labels: List[str]

    if args.task_name in TASK_PRESETS and not (args.task_description and args.class_labels):
        preset = TASK_PRESETS[args.task_name]
        task_description = args.task_description or preset["task_description"]
        if args.class_labels:
            class_labels = [c.strip() for c in args.class_labels.split(",")]
        else:
            class_labels = preset["class_labels"]
    else:
        if not args.task_description:
            raise ValueError(
                f"任务 '{args.task_name}' 不在内置预设中，请通过 --task_description 指定任务描述。"
            )
        if not args.class_labels:
            raise ValueError(
                f"任务 '{args.task_name}' 不在内置预设中，请通过 --class_labels 指定类别标签。"
            )
        task_description = args.task_description
        class_labels = [c.strip() for c in args.class_labels.split(",")]

    # 更新 args 中的 task_description 和 class_labels，便于后续保存 summary
    args.task_description = task_description
    args.class_labels = class_labels

    print(f"\n任务: {args.task_name}")
    print(f"描述: {task_description}")
    print(f"类别: {class_labels}")

    if args.batch_workers < 1:
        raise ValueError("--batch_workers 须 >= 1")
    if args.batch_workers > 1:
        print(f"batch 并行进程数: {args.batch_workers}（注意 API 限速与配额）")

    # 解析各 LLM 的 API 配置
    a_api_base, a_api_key = resolve_api_config(
        args.llm_a_api_base, args.llm_a_api_key, args.api_base, args.api_key
    )
    b_api_base, b_api_key = resolve_api_config(
        args.llm_b_api_base, args.llm_b_api_key, args.api_base, args.api_key
    )

    llm_c_configured = False
    c_api_base: str = ""
    c_api_key: str = ""
    if args.use_adjudicator:
        c_api_base, c_api_key = resolve_api_config(
            args.llm_c_api_base, args.llm_c_api_key, args.api_base, args.api_key
        )
        llm_c_configured = True

    print(f"\nLLM-A: {args.llm_a_model} @ {a_api_base}")
    print(f"LLM-B: {args.llm_b_model} @ {b_api_base}")
    if llm_c_configured:
        print(f"LLM-C: {args.llm_c_model} @ {c_api_base}")

    # 创建输出目录
    output_dir = Path(args.output_dir)
    raw_dir = output_dir / "raw"
    parsed_dir = output_dir / "parsed"
    for d in [output_dir, raw_dir, parsed_dir]:
        d.mkdir(parents=True, exist_ok=True)

    error_log_path = output_dir / "error_log.jsonl"
    progress_path = output_dir / "progress.json"

    # 读取输入 CSV
    print(f"\n读取输入 CSV: {args.input_csv}")
    df = pd.read_csv(args.input_csv, encoding="utf-8")
    print(f"原始 concept 数量: {len(df)}")

    # 应用 start_index 和 limit
    if args.start_index > 0:
        df = df.iloc[args.start_index:].reset_index(drop=True)
        print(f"从 index {args.start_index} 开始，剩余 {len(df)} 个 concepts")

    if args.limit is not None:
        df = df.iloc[: args.limit].reset_index(drop=True)
        print(f"限制处理前 {args.limit} 个 concepts，实际 {len(df)} 个")

    print(f"待处理 concept 数量: {len(df)}")

    # ==== LLM-A 处理 ====
    print(f"\n[LLM-A] 开始处理，模型: {args.llm_a_model}")
    results_a = process_llm_for_all_batches(
        df=df,
        batch_size=args.batch_size,
        task_description=task_description,
        class_labels=class_labels,
        api_base=a_api_base,
        api_key=a_api_key,
        model=args.llm_a_model,
        llm_name="llm_a",
        raw_dir=raw_dir,
        parsed_dir=parsed_dir,
        error_log_path=error_log_path,
        max_retries=args.max_retries,
        sleep_seconds=args.sleep_seconds,
        resume=args.resume,
        debug_llm_parse=args.debug_llm_parse,
        debug_llm_parse_max_chars=args.debug_llm_parse_max_chars,
        batch_workers=args.batch_workers,
    )
    print(f"[LLM-A] 完成，共处理 {len(results_a)} 个 concepts")

    # ==== LLM-B 处理 ====
    print(f"\n[LLM-B] 开始处理，模型: {args.llm_b_model}")
    results_b = process_llm_for_all_batches(
        df=df,
        batch_size=args.batch_size,
        task_description=task_description,
        class_labels=class_labels,
        api_base=b_api_base,
        api_key=b_api_key,
        model=args.llm_b_model,
        llm_name="llm_b",
        raw_dir=raw_dir,
        parsed_dir=parsed_dir,
        error_log_path=error_log_path,
        max_retries=args.max_retries,
        sleep_seconds=args.sleep_seconds,
        resume=args.resume,
        debug_llm_parse=args.debug_llm_parse,
        debug_llm_parse_max_chars=args.debug_llm_parse_max_chars,
        batch_workers=args.batch_workers,
    )
    print(f"[LLM-B] 完成，共处理 {len(results_b)} 个 concepts")

    # ==== Consensus 合并 ====
    print("\n[Consensus] 合并 LLM-A 和 LLM-B 结果...")
    consensus_df = merge_llm_results(
        df_original=df,
        results_a=results_a,
        results_b=results_b,
        class_labels=class_labels,
        score_merge_strategy=args.score_merge_strategy,
    )

    uncertain_count = int((consensus_df["final_decision"] == "uncertain").sum())
    print(f"[Consensus] 完成。keep: {(consensus_df['final_decision']=='keep').sum()}, "
          f"drop: {(consensus_df['final_decision']=='drop').sum()}, "
          f"uncertain: {uncertain_count}")

    # ==== LLM-C Adjudication (可选) ====
    llm_c_results: Optional[List[Dict]] = None

    if args.use_adjudicator and uncertain_count > 0:
        uncertain_df = consensus_df[consensus_df["final_decision"] == "uncertain"].copy()

        # 从 df（原始字段）中获取 uncertain 行
        uncertain_original_df = df[df["concept"].isin(uncertain_df["concept"].tolist())].copy()

        llm_c_results = run_adjudication(
            uncertain_df=uncertain_original_df,
            df_original=df,
            results_a=results_a,
            results_b=results_b,
            class_labels=class_labels,
            task_description=task_description,
            api_base=c_api_base,
            api_key=c_api_key,
            model=args.llm_c_model,
            batch_size=args.batch_size,
            raw_dir=raw_dir,
            parsed_dir=parsed_dir,
            error_log_path=error_log_path,
            max_retries=args.max_retries,
            sleep_seconds=args.sleep_seconds,
            resume=args.resume,
            debug_llm_parse=args.debug_llm_parse,
            debug_llm_parse_max_chars=args.debug_llm_parse_max_chars,
            batch_workers=args.batch_workers,
        )
        print(f"[LLM-C] 完成，裁决了 {len(llm_c_results)} 个 concepts")

    elif args.use_adjudicator and uncertain_count == 0:
        print("[LLM-C] 无 uncertain 样本，跳过 adjudication。")

    # ==== 保存所有输出 ====
    print("\n[输出] 保存所有结果文件...")
    save_outputs(
        consensus_df=consensus_df,
        output_dir=output_dir,
        class_labels=class_labels,
        keep_uncertain=args.keep_uncertain,
        llm_c_results=llm_c_results,
        args=args,
    )

    # 保存 progress
    save_progress(
        progress_path,
        {
            "status": "completed",
            "task_name": args.task_name,
            "processed_count": len(df),
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
    )

    print("\n[完成] 所有任务执行完毕。")


if __name__ == "__main__":
    main()
