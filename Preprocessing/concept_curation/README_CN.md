# SlideKo 概念构建（MGCC）

[English](README.md)

本模块实现 SlideKo 的形态学约束概念筛选流程（Morphology-Grounded Concept Curation，MGCC），将 ROI 层级病理短语依次经过 UMLS 规范化、语义类型过滤、CUI 层级去重、多 LLM 任务相关性判断、专家复核和 TITAN 文本编码，生成任务特异的病理概念池及其文本特征。

## 流程概览

```mermaid
flowchart LR
    A[Quilt-1M ROI 文本] --> B[UMLS 映射]
    B --> C[语义类型过滤]
    C --> D[CUI 层级去重]
    D --> E[LLM-A 与 LLM-B 独立判断]
    E --> F[可选 LLM-C 裁决]
    F --> G[病理专家复核]
    G --> H[任务特异概念池]
    H --> I[TITAN 文本特征]
```

## 目录结构

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

UMLS 和 LLM 中间结果分别生成在本地 `umls_concepts/` 和 `llm_filtered/` 中。适合发布的最终概念池和 TITAN 特征统一保存在 `../../resources/concept_pools/<TASK>/`，供训练与评估代码独立读取。

## 支持的任务

脚本内置四个 WSI 分类任务：

| 任务名 | 分类目标 | 参考概念池大小 |
|---|---|---:|
| `CAMELYON16` | 正常组织与乳腺癌淋巴结转移 | 534 |
| `NSCLC` | LUAD 与 LUSC | 688 |
| `UBC-OCEAN` | 卵巢癌亚型分类 | 490 |
| `TCGA-RCC` | KIRC、KICH 与 KIRP | 483 |

也可以通过 `--task_description` 和 `--class_labels` 定义其他任务。

## 安装

建议创建独立 Python 环境并安装以下依赖：

```bash
pip install pandas tqdm spacy scispacy requests torch transformers
```

另外需要安装与 SciSpaCy 版本兼容的模型。TITAN 通过 Hugging Face 的 `MahmoodLab/TITAN` 加载，并启用 `trust_remote_code=True`；为了保证结果可复现，正式发布时应固定依赖版本和模型 revision。

以下命令均从本目录执行。

## 输入数据

UMLS 阶段接收包含 ROI 文本列的 CSV。该列默认名为 `roi_text`，每个单元格应为可由 Python 解析的病理短语列表：

```text
["lymphocyte", "tumor cell", "necrosis"]
```

ROI 层级病理文本可以来自 Quilt-1M，也可以来自具有相同数据结构的其他语料。Quilt-1M 可以通过以下地址获取：

- [Quilt-1M 官方仓库及受限访问说明](https://github.com/wisdomikezogwo/quilt1m#data-quilt-1m-restricted-access)
- [Google Drive 备用下载地址](https://drive.google.com/file/d/1FkUPQi_DwQKbEubZdhEXJKjY4OFUJfxO/view?usp=drive_link)

Quilt-1M 官方发布要求使用者同意相应的访问和科研使用条款。通过备用地址获取数据前，请确认已获得访问授权，并遵守官方许可、科研用途和再分发要求。

## 步骤 1：UMLS 映射与 CUI 去重

`build_umls_filtered_concepts_v1.py` 完成短语规范化、UMLS 映射、语义类型过滤、频次统计和 CUI 层级同义词去重。

```bash
python build_umls_filtered_concepts_v1.py \
  --csv_path /path/to/quilt_1M_lookup.csv \
  --roi_col roi_text \
  --output_dir ./umls_concepts \
  --spacy_model en_core_sci_sm \
  --umls_threshold 0.80 \
  --batch_size 128
```

PowerShell 用户应将每行末尾的 `\` 替换为反引号。

主要输出：

| 文件 | 说明 |
|---|---|
| `concepts_umls_cui_dedup_semantic_filtered.csv` | 进入任务筛选阶段的去重概念 |
| `concepts_umls_mapped_all.csv` | 成功完成 UMLS 映射的全部概念 |
| `concepts_umls_unmatched.csv` | 未获得有效 UMLS 映射的短语 |
| `concepts_semantic_type_kept.csv` | 语义类型规则保留的概念 |
| `concepts_semantic_type_dropped.csv` | 语义类型规则剔除的概念 |
| `raw_concepts_frequency.csv` | 原始短语频次 |
| `final_concept_list.txt` | 每行一个保留概念 |
| `args.json`、`summary.json` | 运行参数和汇总统计 |

主表包含 `concept`、`cui`、`umls_name`、`umls_types`、`umls_type_names`、`freq`、`umls_score` 和 `dedup_score` 等字段。进入下一阶段前，应检查语义类型保留表、剔除表和未匹配表。

离线运行时，可以通过 `--local_umls_linker_dir` 指定完整的本地 SciSpaCy UMLS linker 目录。

## 步骤 2：多 LLM 任务相关性筛选

`llm_task_concept_filter.py` 使用 LLM-A 和 LLM-B 独立判断每个概念，并将结果合并为 `keep`、`drop` 或 `uncertain`；启用 `--use_adjudicator` 后，LLM-C 会进一步裁决存在分歧的概念。

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

主要输出：

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

使用 `--resume` 可以从中断位置继续运行。如果模型响应被截断或缺少条目，应减小 `--batch_size`；`--batch_workers` 不宜设置过大，以免触发服务商限流。发布概念池时，应记录服务商、准确的模型版本、运行日期、提示词配置和裁决规则。

## 步骤 3：病理专家复核

对入选概念逐项检查任务相关性、H&E 可观察性、病理合理性、冗余和歧义。推荐使用以下复核表结构：

```text
concept,task,pre_review_decision,expert_decision,reason,reviewer_role,review_date
```

最终任务概念池仅保留专家确认的条目，同时保留复核前决策和复核理由，以支持结果追溯和后续更新。

将专家确认后的表格保存为 `../../resources/concept_pools/<TASK>/final_task_concept_pool.csv`。

## 步骤 4：TITAN 文本特征

`extract_titan_concept_features.py` 使用 TITAN 文本编码器对最终概念进行编码，并保存特征与概念之间的对应关系。

```bash
python extract_titan_concept_features.py \
  --task_name CAMELYON16 \
  --input_root ../../resources/concept_pools \
  --model_name MahmoodLab/TITAN \
  --prompt_mode pathology_templates \
  --batch_size 128
```

输出：

```text
../../resources/concept_pools/<TASK>/titan_features/
├── *_embeddings.pt
├── *_metadata.csv
└── *_summary.json
```

脚本默认套用 7 个病理提示模板，对所得特征取平均并进行 L2 归一化。使用 `--prompt_mode none` 可以直接编码概念，使用 `--no_normalize` 可以关闭归一化。metadata CSV 保存特征行与概念的对应关系，应与 embedding 张量一同发布。

可选的 `--include_patch_description` 模式会将 `patch_visual_description_20x` 列与概念拼接。使用该模式时，应提供 `final_task_concept_pool_patch_desc.csv`，或者通过 `--input_name` 指定其他输入文件，并通过 `--description_column` 指定描述列。

## 可复现性检查

- 确认每个保留概念都能追溯到原始数据行和任务筛选决策。
- 审查 UMLS 未匹配结果和按语义类型剔除的概念。
- 确认最终概念池中的每条记录均有专家复核结果。
- 确认概念池、metadata 和 embedding 的行数完全一致。
- 检查所有 embedding 张量均为有限值且特征维度一致。
- 保存输入语料、最终概念池、metadata 和 embedding 的校验和。
- 记录 Python 依赖、UMLS 资源、模型 revision、提示词、随机种子和 API 运行日期。

## 引用

如果本模块用于论文研究，请根据实际使用情况引用 SlideKo、Quilt-1M、UMLS、SciSpaCy 和 TITAN。
