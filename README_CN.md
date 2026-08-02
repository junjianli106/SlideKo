# SlideKo：面向少样本全切片图像分类的切片自适应知识锚定方法

[![English README](https://img.shields.io/badge/README-English-2b6cb0)](README.md)

本仓库提供论文 **SlideKo: Slide-Adaptive Knowledge Grounding in Multiple Instance Learning for Few-Shot Whole Slide Image Classification** 的 PyTorch 实现。

SlideKo 是一个面向少样本全切片图像（whole slide image，WSI）分类的知识引导多实例学习框架。该方法首先构建任务特异性病理概念池，再根据每张切片的形态特征实例化与当前切片相关的知识，并在训练过程中对齐视觉表示与语义表示。本文在 CAMELYON16、TCGA-NSCLC、TCGA-RCC 和 UBC-OCEAN 上进行了 4-shot、8-shot 和 16-shot 实验。

## 方法

SlideKo 包含四个主要组成部分：

- **形态学引导的概念整理（Morphology-Grounded Concept Curation，MGCC）**：通过 UMLS 标准化、语义类型过滤、多大语言模型相关性评估、专家审核和 TITAN 文本编码，从 ROI 级病理文本中构建任务特异性概念池。
- **切片自适应知识实例化（Slide-Adaptive Knowledge Instantiation，SAKI）**：对每张切片的图像块进行聚类，并从任务概念池中检索与当前切片最相关的病理概念，使知识表示能够适应切片自身的形态特征。
- **双流交叉注意力（Dual-Stream Cross-Attention，DSCA）**：使用类别感知文本查询分别聚合视觉证据和实例化知识，随后融合两个分支以完成分类。
- **邻域引导的语义一致性（Neighborhood-Guided Semantic Consistency，NGSC）**：促进视觉空间和语义空间之间的局部一致性。公式（14）的温度固定为 `0.5`，TITAN 文本编码器参与联合训练，不进行冻结。

发布配置采用论文的主要实验参数：16 个可学习上下文 token、10 个切片聚类、每个聚类选择 10 个概念、概念注意力温度 `0.1`、NGSC 损失权重 `0.1`，邻居数量为 5。

## 目录结构

```text
SlideKo/
├── Cls/                              # 各数据集的 SlideKo 配置
│   ├── CAMELYON16/SlideKo/
│   ├── NSCLC/SlideKo/
│   ├── RCC/SlideKo/
│   └── UBC/SlideKo/
├── datasets/                         # 少样本 WSI 数据集与数据接口
├── models/                           # SlideKo 模型及训练、评估接口
├── Preprocessing/concept_curation/   # MGCC 概念整理流程
├── resources/
│   ├── concept_pools/                # 最终概念池及 TITAN 概念特征
│   └── text_prompt/Claude/           # Claude 生成的类别描述
├── splits/4foldcls/                  # 四折划分及少样本采样文件
├── scripts/Cls/                      # 实验启动脚本
└── train.py                          # 训练与评估入口
```

## 环境安装

建议使用 Python 3.10 创建独立 Conda 环境，并根据本机驱动安装对应的 CUDA 版本 PyTorch：

```bash
conda create -n slideko python=3.10 -y
conda activate slideko

# 以下为 CUDA 12.1 示例，请根据运行平台选择相应的 PyTorch 安装命令。
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install pytorch-lightning transformers pandas numpy scipy scikit-learn \
  pyyaml addict setproctitle colorama tensorboard opencv-python pytorch-toolbelt

# 安装与本机 CUDA 环境兼容的 GPU 版 FAISS。
conda install -c pytorch faiss-gpu -y
```

当前实现使用 GPU FAISS 完成切片级图像块聚类，因此需要支持 CUDA 的运行环境。

## 数据准备

### 1. 下载 WSI 数据集

实验使用以下公开科研数据集。请从官方渠道下载，并遵守相应的数据访问、许可和使用要求。

| 数据集 | 分类任务 | 类别数 | 配置文件 |
|---|---|---:|---|
| [CAMELYON16](https://camelyon16.grand-challenge.org/Home/) | 正常组织与乳腺癌淋巴结转移分类 | 2 | [`Cls/CAMELYON16/SlideKo/SlideKo.yaml`](Cls/CAMELYON16/SlideKo/SlideKo.yaml) |
| [TCGA-NSCLC](https://portal.gdc.cancer.gov/) | LUAD 与 LUSC 分类 | 2 | [`Cls/NSCLC/SlideKo/SlideKo.yaml`](Cls/NSCLC/SlideKo/SlideKo.yaml) |
| [TCGA-RCC](https://portal.gdc.cancer.gov/) | KIRC、KICH 与 KIRP 分类 | 3 | [`Cls/RCC/SlideKo/SlideKo.yaml`](Cls/RCC/SlideKo/SlideKo.yaml) |
| [UBC-OCEAN](https://www.kaggle.com/competitions/UBC-OCEAN) | 卵巢癌亚型分类 | 5 | [`Cls/UBC/SlideKo/SlideKo.yaml`](Cls/UBC/SlideKo/SlideKo.yaml) |

### 2. 准备图像块特征

按照论文设置，在 `20×` 放大倍率下提取空间尺寸为 `448 × 448` 像素的组织图像块，并将其编码为 768 维、与 TITAN 对应的图像块特征。每张切片保存为一个 PyTorch 张量：

```text
/path/to/features/
├── slide_001.pt
├── slide_002.pt
└── ...
```

每个张量的形状应为 `[N, 768]`，其中 `N` 表示该切片的图像块数量；文件名必须与相应数据划分 CSV 中的 `slide_id` 一致。请将所选 YAML 文件中的 `Data.data_high_dir` 和 `Data.data_low_dir` 设置为特征目录。数据加载器会将切片特征预加载到内存，因此运行机器需要具备足够的内存容量。

### 3. 配置 TITAN

代码通过 Hugging Face 加载 TITAN，并设置 `trust_remote_code=True`。请先申请访问 [`MahmoodLab/TITAN`](https://huggingface.co/MahmoodLab/TITAN)，并遵守其许可与访问条件。可以选择在线加载或从本地快照加载：

```yaml
# 通过 Hugging Face 下载或加载
Model:
    titan_model_path: MahmoodLab/TITAN
    local_files_only: False
```

```yaml
# 加载已有的本地快照
Model:
    titan_model_path: /path/to/local/TITAN/snapshot
    local_files_only: True
```

当 `titan_model_path` 被省略或设置为 `null` 时，也可以通过 `TITAN_MODEL_PATH` 环境变量指定模型位置。设置 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1` 或 `SLIDEKO_HF_LOCAL_FILES_ONLY=1` 会强制采用仅本地加载模式。

### 4. 配置运行主机

首次运行前，需要在 [`train.py`](train.py) 的 `host_data_mapping` 中加入当前运行环境的主机名或用户主目录。现有启动程序会检查运行主机，并在主机未注册时停止执行。YAML 中的特征目录应与该映射保持一致。

### 5. 构建任务特异性概念池

MGCC 将大规模病理词汇整理为面向具体分类任务的紧凑概念池。仓库中发布的概念池按照以下过程生成：

1. **ROI 文本收集：**从 Quilt-1M 的 ROI 级描述中收集病理形态学短语。
2. **UMLS 标准化：**将短语映射到 UMLS 概念，保留与病理相关的语义类型，依据概念唯一标识符（Concept Unique Identifier，CUI）合并同义词，并去除无法匹配或不适用的术语。
3. **任务相关性评估：**使用两个大语言模型分别判断每个概念是否与目标分类任务相关，以及该概念是否能够通过 H&E 形态观察；存在分歧的概念可以交由第三个大语言模型进行裁决。
4. **专家审核：**从病理相关性、H&E 可观察性、冗余性、术语歧义和类别信息泄漏等方面审核保留项及不确定项，只将专家批准的概念纳入最终概念池。
5. **TITAN 编码：**使用病理提示模板对每个批准的概念进行编码，对多个模板产生的 TITAN 文本特征求平均并进行 L2 归一化，同时在元数据中保存概念与特征行之间的严格对应关系。

| 任务概念池 | 分类目标 | 概念数量 | 发布目录 |
|---|---|---:|---|
| CAMELYON16 | 正常组织与乳腺癌转移 | 534 | [`resources/concept_pools/CAMELYON16/`](resources/concept_pools/CAMELYON16/) |
| NSCLC | LUAD 与 LUSC | 688 | [`resources/concept_pools/NSCLC/`](resources/concept_pools/NSCLC/) |
| TCGA-RCC | KIRC、KICH 与 KIRP | 483 | [`resources/concept_pools/TCGA-RCC/`](resources/concept_pools/TCGA-RCC/) |
| UBC-OCEAN | 卵巢癌亚型分类 | 490 | [`resources/concept_pools/UBC-OCEAN/`](resources/concept_pools/UBC-OCEAN/) |

每个发布目录均包含经过专家审核的 `final_task_concept_pool.csv`，以及保存编码张量、概念元数据和提取摘要的 `titan_features/` 目录。SlideKo 根据数据集 YAML 中的 `Data.concept_dir` 加载相应的概念特征张量，并在每次前向传播过程中通过 SAKI 选择与当前切片相关的概念。因此，该方法在概念池层面具有任务特异性，在训练和推理时又具有切片自适应性。

如需复现或扩展该流程，请参阅 [MGCC 中文预处理说明](Preprocessing/concept_curation/README_CN.md)或 [MGCC 英文预处理说明](Preprocessing/concept_curation/README.md)。ROI 级病理文本可以从 [Quilt-1M 官方发布页面](https://github.com/wisdomikezogwo/quilt1m#data-quilt-1m-restricted-access)或[谷歌云盘地址](https://drive.google.com/file/d/1FkUPQi_DwQKbEubZdhEXJKjY4OFUJfxO/view?usp=drive_link)获取。无论使用哪个地址，都应遵守 Quilt-1M 官方许可、科研用途和数据再分发要求。

### 6. 使用已发布的数据划分和知识资源

四折数据划分和固定少样本采样文件位于 [`splits/4foldcls/`](splits/4foldcls/)。任务特异性概念池、TITAN 概念特征及其元数据位于 [`resources/concept_pools/`](resources/concept_pools/)，SlideKo 使用的类别描述位于 [`resources/text_prompt/Claude/`](resources/text_prompt/Claude/)。

## 训练

请在 `SlideKo` 目录下运行命令。以下示例训练 CAMELYON16 的第 0 折 16-shot 实验：

```bash
python train.py \
  --stage=train \
  --config=Cls/CAMELYON16/SlideKo/SlideKo.yaml \
  --gpus=0 \
  --fold=0 \
  --seed=1024 \
  --n_shot=16
```

运行全部四折实验：

```bash
for fold in 0 1 2 3; do
  python train.py --stage=train --config=Cls/CAMELYON16/SlideKo/SlideKo.yaml \
    --gpus=0 --fold=${fold} --seed=2025 --n_shot=16
done
```


其他数据集只需替换配置文件路径：

```bash
# TCGA-NSCLC
python train.py --stage=train --config=Cls/NSCLC/SlideKo/SlideKo.yaml --gpus=0 --fold=0 --seed=2025 --n_shot=16

# TCGA-RCC
python train.py --stage=train --config=Cls/RCC/SlideKo/SlideKo.yaml --gpus=0 --fold=0 --seed=2025 --n_shot=16

# UBC-OCEAN
python train.py --stage=train --config=Cls/UBC/SlideKo/SlideKo.yaml --gpus=0 --fold=0 --seed=2025 --n_shot=16
```

## 评估

将运行阶段修改为 `test`，并保持数据集、折数、随机种子和 shot 设置与训练阶段一致，即可评估对应实验目录中的模型检查点：

```bash
python train.py \
  --stage=test \
  --config=Cls/CAMELYON16/SlideKo/SlideKo.yaml \
  --gpus=0 \
  --fold=0 \
  --seed=1024 \
  --n_shot=16
```

默认情况下，模型检查点、TensorBoard/CSV 日志、评估指标、预测概率、标签和 logits 保存到：

```text
Journal_logs/Cls/<DATASET>/SlideKo/SlideKo/Shot<N>/fold<FOLD>/
```

主要评估文件包括 `test_metrics.csv`、`all_probs.csv`、`all_labels.csv`、`all_logits.csv` 及对应的压缩 NumPy 文件。如果某个实验目录中已经存在 `test_metrics.csv`，启动程序会将该实验视为已完成并跳过运行；如需重新运行相同设置，请先移走或归档该文件。


## 引用

如果本工作对您的研究有所帮助，请引用：

```bibtex
@inproceedings{li2026universal,
  title={Universal-to-Specific: Dynamic Knowledge-Guided Multiple Instance Learning for Few-Shot Whole Slide Image Classification},
  author={Li, Junjian and Kuang, Hulin and Liu, Jin and Yue, Hailin and He, Mengshen and Wang, Jianxin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={26614--26623},
  year={2026}
}
```
