# SlideKo Concept Curation (MGCC)

[中文说明](README_CN.md)

This module implements the Morphology-Grounded Concept Curation (MGCC) pipeline used by SlideKo. It converts ROI-level pathology phrases into task-specific concept pools through UMLS normalization, semantic-type filtering, CUI-level deduplication, multi-LLM relevance assessment, expert review, and TITAN text encoding.

## Overview

```mermaid
flowchart LR
    A[Quilt-1M ROI texts] --> B[UMLS mapping]
    B --> C[Semantic-type filtering]
    C --> D[CUI-level deduplication]
    D --> E[LLM-A and LLM-B assessment]
    E --> F[Optional LLM-C adjudication]
    F --> G[Expert review]
    G --> H[Task-specific concept pool]
    H --> I[TITAN text embeddings]
```

## Directory structure

```text
SlideKo/
├── Preprocessing/
│   └── concept_curation/
│       ├── README.md
│       ├── README_CN.md
│       ├── build_umls_filtered_concepts_v1.py
│       ├── llm_task_concept_filter.py
│       └── extract_titan_concept_features.py
└── resources/
    └── concept_pools/
        ├── CAMELYON16/
        ├── NSCLC/
        ├── TCGA-RCC/
        └── UBC-OCEAN/
```

UMLS and LLM intermediate results are generated locally under `umls_concepts/` and `llm_filtered/`. Release-ready concept pools and TITAN features are stored under `../../resources/concept_pools/<TASK>/` so that training and evaluation code can access them independently of the preprocessing workspace.

## Supported tasks

The built-in task presets cover four WSI classification datasets:

| Task name | Classification target | Reference pool size |
|---|---|---:|
| `CAMELYON16` | Normal tissue vs breast cancer metastasis | 534 |
| `NSCLC` | LUAD vs LUSC | 688 |
| `UBC-OCEAN` | Ovarian cancer subtyping | 490 |
| `TCGA-RCC` | KIRC vs KICH vs KIRP | 483 |

Custom tasks can be defined with `--task_description` and `--class_labels`.

## Installation

Create an isolated Python environment and install the required packages:

```bash
pip install pandas tqdm spacy scispacy requests torch transformers
```

Install a SciSpaCy model compatible with the selected SciSpaCy release. TITAN is loaded from `MahmoodLab/TITAN` through Hugging Face with `trust_remote_code=True`; for reproducible experiments, pin the package versions and model revision in the release environment.

Run all commands from this directory. `python` can be replaced by `uv run python` or the equivalent command in a Conda environment.

## Input data

The UMLS stage accepts a CSV containing an ROI-text column. By default, the column is named `roi_text`, and each cell is a Python-literal list of candidate pathology phrases:

```text
["lymphocyte", "tumor cell", "necrosis"]
```

ROI-level pathology text can be obtained from Quilt-1M or another corpus with an equivalent schema. Quilt-1M can be accessed through either of the following links:

- [Official Quilt-1M repository and restricted-access instructions](https://github.com/wisdomikezogwo/quilt1m#data-quilt-1m-restricted-access)
- [Alternative Google Drive link](https://drive.google.com/file/d/1FkUPQi_DwQKbEubZdhEXJKjY4OFUJfxO/view?usp=drive_link)

The official Quilt-1M release requires users to agree to its access and research-use terms. Before using the alternative link, confirm that you are authorized to access the data and comply with the official license, research-use, and redistribution requirements.

## Step 1: UMLS mapping and CUI deduplication

`build_umls_filtered_concepts_v1.py` performs phrase normalization, UMLS mapping, semantic-type filtering, frequency analysis, and CUI-level synonym deduplication.

```bash
python build_umls_filtered_concepts_v1.py \
  --csv_path /path/to/quilt_1M_lookup.csv \
  --roi_col roi_text \
  --output_dir ./umls_concepts \
  --spacy_model en_core_sci_sm \
  --umls_threshold 0.80 \
  --batch_size 128
```

PowerShell users should replace each trailing `\` with a backtick.

Main outputs:

| File | Description |
|---|---|
| `concepts_umls_cui_dedup_semantic_filtered.csv` | Deduplicated concepts retained for task filtering |
| `concepts_umls_mapped_all.csv` | All successfully mapped concepts |
| `concepts_umls_unmatched.csv` | Phrases without a valid UMLS match |
| `concepts_semantic_type_kept.csv` | Concepts retained by semantic-type rules |
| `concepts_semantic_type_dropped.csv` | Concepts removed by semantic-type rules |
| `raw_concepts_frequency.csv` | Raw phrase frequencies |
| `final_concept_list.txt` | One retained concept per line |
| `args.json`, `summary.json` | Run configuration and summary statistics |

The primary table contains `concept`, `cui`, `umls_name`, `umls_types`, `umls_type_names`, `freq`, `umls_score`, and `dedup_score`. Review the kept, dropped, and unmatched tables before task-specific filtering.

For offline execution, pass `--local_umls_linker_dir` with a complete local SciSpaCy UMLS linker directory.

## Step 2: Multi-LLM task relevance filtering

`llm_task_concept_filter.py` asks LLM-A and LLM-B to independently assess each concept. Their results are merged into `keep`, `drop`, or `uncertain`; `--use_adjudicator` enables LLM-C to resolve disputed concepts.

```bash
python llm_task_concept_filter.py \
  --input_csv ./umls_concepts/concepts_umls_cui_dedup_semantic_filtered.csv \
  --output_dir ./llm_filtered/CAMELYON16 \
  --task_name CAMELYON16 \
  --llm_a_api_base https://provider-a.example/v1 \
  --llm_a_api_key "$LLM_A_API_KEY" \
  --llm_a_model MODEL_A \
  --llm_b_api_base https://provider-b.example/v1 \
  --llm_b_api_key "$LLM_B_API_KEY" \
  --llm_b_model MODEL_B \
  --use_adjudicator \
  --llm_c_api_base https://provider-c.example/v1 \
  --llm_c_api_key "$LLM_C_API_KEY" \
  --llm_c_model MODEL_C \
  --batch_size 50 \
  --score_merge_strategy mean \
  --resume
```

Main outputs:

```text
llm_filtered/<TASK>/
├── raw/
├── parsed/
├── llm_a_results.csv
├── llm_b_results.csv
├── consensus_all.csv
├── consensus_keep.csv
├── consensus_drop.csv
├── consensus_uncertain.csv
├── llm_c_adjudication_results.csv
├── consensus_after_adjudication.csv
├── final_task_concept_pool.csv
├── final_concept_list.txt
└── summary.json
```

Use `--resume` to continue an interrupted run. Reduce `--batch_size` when responses are truncated or incomplete, and keep `--batch_workers` conservative to avoid provider rate limits. Record the provider, exact model version, date, prompt configuration, and adjudication policy for every released concept pool.

## Step 3: Expert review

Review the selected concepts for task relevance, H&E observability, pathological plausibility, redundancy, and ambiguous terminology. A review table can use the following schema:

```text
concept,task,pre_review_decision,expert_decision,reason,reviewer_role,review_date
```

Use only expert-approved rows as the final task-specific concept pool. Preserve the pre-review decision and review rationale to support provenance and future updates.

Save the approved table as `../../resources/concept_pools/<TASK>/final_task_concept_pool.csv`.

## Step 4: TITAN text embeddings

`extract_titan_concept_features.py` encodes the final concepts with the TITAN text encoder and stores the embedding-to-concept correspondence.

```bash
python extract_titan_concept_features.py \
  --task_name CAMELYON16 \
  --input_root ../../resources/concept_pools \
  --model_name MahmoodLab/TITAN \
  --prompt_mode pathology_templates \
  --batch_size 128
```

Outputs:

```text
../../resources/concept_pools/<TASK>/titan_features/
├── *_embeddings.pt
├── *_metadata.csv
└── *_summary.json
```

By default, the script applies seven pathology prompt templates, averages the resulting embeddings, and performs L2 normalization. Use `--prompt_mode none` for direct concept encoding and `--no_normalize` to disable normalization. The metadata CSV preserves row-to-concept correspondence and should be distributed with every embedding tensor.

The optional `--include_patch_description` mode concatenates the `patch_visual_description_20x` column with each concept. When using this mode, provide `final_task_concept_pool_patch_desc.csv` or select another input file with `--input_name` and another description column with `--description_column`.

## Reproducibility checks

- Verify that each retained concept has a valid source row and task decision.
- Audit UMLS unmatched results and semantic-type exclusions.
- Confirm that expert decisions are recorded for every final concept.
- Confirm that concept-pool, metadata, and embedding row counts are identical.
- Check that all embedding tensors are finite and have a consistent feature dimension.
- Save checksums for the input corpus, final concept pools, metadata, and embeddings.
- Record Python dependencies, UMLS resources, model revisions, prompts, seeds, and API execution dates.

## Security

Do not commit API keys, access tokens, private endpoints, or provider response files containing sensitive information. Supply credentials through environment variables or a local configuration file excluded by `.gitignore`, and rotate any credential that has previously been stored in source code.

## Citation

If this module supports a publication, cite SlideKo together with Quilt-1M, UMLS, SciSpaCy, and TITAN as appropriate.
