# SlideKo: Slide-Adaptive Knowledge Grounding for Few-Shot WSI Classification

[![中文 README](https://img.shields.io/badge/README-中文-d7191c)](README_CN.md)

PyTorch implementation of **SlideKo: Slide-Adaptive Knowledge Grounding in Multiple Instance Learning for Few-Shot Whole Slide Image Classification**.

SlideKo is a knowledge-grounded multiple instance learning framework for few-shot whole slide image (WSI) classification. It constructs task-specific pathology concept pools, instantiates slide-relevant knowledge from those pools, and aligns visual and semantic representations during training. The framework is evaluated under 4-, 8-, and 16-shot settings on CAMELYON16, TCGA-NSCLC, TCGA-RCC, and UBC-OCEAN.

## Method

SlideKo contains four main components:

- **Morphology-Grounded Concept Curation (MGCC)** builds task-specific concept pools from ROI-level pathology text through UMLS normalization, semantic filtering, multi-LLM relevance assessment, expert review, and TITAN text encoding.
- **Slide-Adaptive Knowledge Instantiation (SAKI)** clusters the patches of each slide and retrieves the most relevant pathology concepts, allowing the knowledge representation to adapt to the morphology of the current slide.
- **Dual-Stream Cross-Attention (DSCA)** separately aggregates visual evidence and instantiated knowledge with class-aware text queries, then combines the two streams for classification.
- **Neighborhood-Guided Semantic Consistency (NGSC)** promotes local consistency between the visual and semantic spaces. The temperature in Eq. (14) is fixed to `0.5`, and the TITAN text encoder remains trainable.

The released configuration follows the main paper setting: 16 learnable context tokens, 10 slide clusters, 10 concepts selected per cluster, concept-attention temperature `0.1`, NGSC weight `0.1`, and 5 neighbors.

## Repository structure

```text
SlideKo/
├── Cls/                              # Dataset-specific SlideKo configurations
│   ├── CAMELYON16/SlideKo/
│   ├── NSCLC/SlideKo/
│   ├── RCC/SlideKo/
│   └── UBC/SlideKo/
├── datasets/                         # Few-shot WSI dataset and data interface
├── models/                           # SlideKo and training/evaluation interfaces
├── Preprocessing/concept_curation/   # MGCC concept-curation pipeline
├── resources/
│   ├── concept_pools/                # Final concepts and TITAN concept embeddings
│   └── text_prompt/Claude/           # Claude-generated class descriptions
├── splits/4foldcls/                  # Four-fold splits and few-shot samples
├── scripts/Cls/                      # Experiment-launching utilities
└── train.py                          # Training and evaluation entry point
```

## Installation

We recommend using a dedicated Conda environment with Python 3.10 and a CUDA-enabled PyTorch installation compatible with the local driver:

```bash
conda create -n slideko python=3.10 -y
conda activate slideko

# Example for CUDA 12.1; select the appropriate command for your platform.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install pytorch-lightning transformers pandas numpy scipy scikit-learn \
  pyyaml addict setproctitle colorama tensorboard opencv-python pytorch-toolbelt

# Install a GPU-enabled FAISS build compatible with your CUDA environment.
conda install -c pytorch faiss-gpu -y
```

SlideKo uses GPU FAISS for slide-level patch clustering, so a CUDA-capable environment is required by the current implementation.

## Data preparation

### 1. Download the WSI datasets

The experiments use the following public research datasets. Download them from their official providers and comply with the corresponding access, licensing, and usage requirements.

| Dataset | Task | Classes | Configuration |
|---|---|---:|---|
| [CAMELYON16](https://camelyon16.grand-challenge.org/Home/) | Lymph-node metastasis detection | 2 | [`Cls/CAMELYON16/SlideKo/SlideKo.yaml`](Cls/CAMELYON16/SlideKo/SlideKo.yaml) |
| [TCGA-NSCLC](https://portal.gdc.cancer.gov/) | LUAD vs LUSC | 2 | [`Cls/NSCLC/SlideKo/SlideKo.yaml`](Cls/NSCLC/SlideKo/SlideKo.yaml) |
| [TCGA-RCC](https://portal.gdc.cancer.gov/) | KIRC vs KICH vs KIRP | 3 | [`Cls/RCC/SlideKo/SlideKo.yaml`](Cls/RCC/SlideKo/SlideKo.yaml) |
| [UBC-OCEAN](https://www.kaggle.com/competitions/UBC-OCEAN) | Ovarian cancer subtype classification | 5 | [`Cls/UBC/SlideKo/SlideKo.yaml`](Cls/UBC/SlideKo/SlideKo.yaml) |

### 2. Prepare patch features

Following the paper, tissue patches are extracted at `20×` magnification with a spatial size of `448 × 448` pixels and encoded as 768-dimensional TITAN-compatible features. Save one PyTorch tensor per slide:

```text
/path/to/features/
├── slide_001.pt
├── slide_002.pt
└── ...
```

Each tensor should have shape `[N, 768]`, where `N` is the number of patches in the slide, and its filename must match the `slide_id` in the corresponding split CSV. Set `Data.data_high_dir` and `Data.data_low_dir` in the selected YAML file to the feature directory. The dataset loader preloads slide features into memory, so ensure that the host has sufficient RAM.

### 3. Configure TITAN

TITAN is loaded through Hugging Face with `trust_remote_code=True`. Request access to [`MahmoodLab/TITAN`](https://huggingface.co/MahmoodLab/TITAN) and follow its license and access conditions. Choose one of the following configurations:

```yaml
# Download/load through Hugging Face
Model:
    titan_model_path: MahmoodLab/TITAN
    local_files_only: False
```

```yaml
# Load an existing local snapshot
Model:
    titan_model_path: /path/to/local/TITAN/snapshot
    local_files_only: True
```

If `titan_model_path` is omitted or set to `null`, the model path can instead be supplied through the `TITAN_MODEL_PATH` environment variable. `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, or `SLIDEKO_HF_LOCAL_FILES_ONLY=1` forces local-only loading.

### 4. Configure the execution host

Before the first run, update `host_data_mapping` in [`train.py`](train.py) with the hostname or home directory of the execution environment. The current launcher validates the host and stops on an unregistered machine. Keep the feature paths in the YAML files consistent with this mapping.

### 5. Build task-specific concept pools

MGCC converts a large pathology vocabulary into a compact concept pool tailored to each classification task. The released pools were produced with the following workflow:

1. **ROI-text collection:** collect pathology phrases from the ROI-level descriptions in Quilt-1M.
2. **UMLS normalization:** map phrases to UMLS concepts, retain pathology-relevant semantic types, merge synonyms by Concept Unique Identifier (CUI), and remove unmatched or unsuitable terms.
3. **Task-relevance assessment:** ask two LLMs to independently determine whether each concept is relevant to the target classification task and observable in H&E morphology; disputed concepts can be resolved by a third adjudicating LLM.
4. **Expert review:** review retained and uncertain concepts for pathological relevance, H&E observability, redundancy, ambiguity, and class leakage, retaining only expert-approved concepts.
5. **TITAN encoding:** encode every approved concept with pathology prompt templates, average and L2-normalize its TITAN text embeddings, and preserve the exact concept-to-row correspondence in the metadata.

| Task-specific pool | Classification target | Concepts | Released directory |
|---|---|---:|---|
| CAMELYON16 | Normal tissue vs breast cancer metastasis | 534 | [`resources/concept_pools/CAMELYON16/`](resources/concept_pools/CAMELYON16/) |
| NSCLC | LUAD vs LUSC | 688 | [`resources/concept_pools/NSCLC/`](resources/concept_pools/NSCLC/) |
| TCGA-RCC | KIRC vs KICH vs KIRP | 483 | [`resources/concept_pools/TCGA-RCC/`](resources/concept_pools/TCGA-RCC/) |
| UBC-OCEAN | Ovarian cancer subtype classification | 490 | [`resources/concept_pools/UBC-OCEAN/`](resources/concept_pools/UBC-OCEAN/) |

Each released directory contains the expert-reviewed `final_task_concept_pool.csv` and a `titan_features/` directory containing the encoded tensor, concept metadata, and extraction summary. SlideKo loads the embedding tensor specified by `Data.concept_dir` in the dataset YAML and applies SAKI to select slide-relevant concepts during each forward pass. Concept selection is therefore task-specific at the pool level and slide-adaptive at inference and training time.

To reproduce or extend this process, follow the [English MGCC preprocessing guide](Preprocessing/concept_curation/README.md) or the [Chinese MGCC preprocessing guide](Preprocessing/concept_curation/README_CN.md). ROI-level pathology text can be obtained from the [official Quilt-1M release](https://github.com/wisdomikezogwo/quilt1m#data-quilt-1m-restricted-access) or the [Google Drive mirror](https://drive.google.com/file/d/1FkUPQi_DwQKbEubZdhEXJKjY4OFUJfxO/view?usp=drive_link). Accessing either source remains subject to the official Quilt-1M license, research-use terms, and redistribution requirements.

### 6. Use the released splits and knowledge resources

Four-fold dataset splits and fixed few-shot samples are provided under [`splits/4foldcls/`](splits/4foldcls/). The released task-specific concept pools, TITAN concept embeddings, and metadata are stored under [`resources/concept_pools/`](resources/concept_pools/), while the class descriptions used by SlideKo are stored under [`resources/text_prompt/Claude/`](resources/text_prompt/Claude/).

## Training

Run commands from the `SlideKo` directory. The following example trains CAMELYON16 fold 0 in the 16-shot setting:

```bash
python train.py \
  --stage=train \
  --config=Cls/CAMELYON16/SlideKo/SlideKo.yaml \
  --gpus=0 \
  --fold=0 \
  --seed=1024 \
  --n_shot=16
```

To run all four folds:

```bash
for fold in 0 1 2 3; do
  python train.py --stage=train --config=Cls/CAMELYON16/SlideKo/SlideKo.yaml \
    --gpus=0 --fold=${fold} --seed=2025 --n_shot=16
done
```


The other datasets can be trained by replacing the configuration path:

```bash
# TCGA-NSCLC
python train.py --stage=train --config=Cls/NSCLC/SlideKo/SlideKo.yaml --gpus=0 --fold=0 --seed=2025 --n_shot=16

# TCGA-RCC
python train.py --stage=train --config=Cls/RCC/SlideKo/SlideKo.yaml --gpus=0 --fold=0 --seed=2025 --n_shot=16

# UBC-OCEAN
python train.py --stage=train --config=Cls/UBC/SlideKo/SlideKo.yaml --gpus=0 --fold=0 --seed=2025 --n_shot=16
```

## Evaluation

Evaluate the checkpoints in the corresponding experiment directory by changing the stage to `test` while retaining the same dataset, fold, seed, and shot setting:

```bash
python train.py \
  --stage=test \
  --config=Cls/CAMELYON16/SlideKo/SlideKo.yaml \
  --gpus=0 \
  --fold=0 \
  --seed=1024 \
  --n_shot=16
```

By default, checkpoints, TensorBoard/CSV logs, metrics, probabilities, labels, and logits are written to:

```text
Journal_logs/Cls/<DATASET>/SlideKo/SlideKo/Shot<N>/fold<FOLD>/
```

The principal evaluation files are `test_metrics.csv`, `all_probs.csv`, `all_labels.csv`, `all_logits.csv`, and their compressed NumPy counterparts. If `test_metrics.csv` already exists for an experiment, the launcher treats the run as completed and skips it; remove or archive that file before intentionally rerunning the same setting.


## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{li2026universal,
  title={Universal-to-Specific: Dynamic Knowledge-Guided Multiple Instance Learning for Few-Shot Whole Slide Image Classification},
  author={Li, Junjian and Kuang, Hulin and Liu, Jin and Yue, Hailin and He, Mengshen and Wang, Jianxin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={26614--26623},
  year={2026}
}

```
