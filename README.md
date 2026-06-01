# LTN-CNF-Axiom-Classification

> **神经符号整合（Neuro-Symbolic Integration）复现与改进实验**  
> 基于 Logic Tensor Networks 框架，将一阶逻辑规则与深度学习结合，  
> 对 CNF 格式数学公理进行多标签分类。

---

## 📌 项目简介

本项目复现并改进了论文
**LTNtorch: PyTorch Implementation of Logic Tensor Networks**
（Carraro et al., 2024），
将其应用于 CNF（合取范式）数学公理数据集的 **7 谓词多标签分类任务**。

同时包含另一篇相关论文 **SBR（Semantic Based Regularization）** 的参考材料。

### 原论文做了什么？

LTNtorch 提出了一种**神经符号整合框架**，核心思想是：

- 用 **Real Logic** 一阶语言定义逻辑知识库（一组规则）
- 通过 **Grounding（G）机制** 将逻辑公式映射为可微分的计算图
- 采用 **模糊逻辑语义（Product 配置）** 使所有逻辑算子全程可微
- 通过梯度下降**最大化知识库满足度**，同时学习神经网络参数

原论文以猫狗二分类为示例，仅有 1 个谓词、2 条规则、完全均衡的数据集。

### 本实验相较于原论文做了什么改进？

| 改进点 | 原论文 | 本实验 |
|--------|--------|--------|
| 任务规模 | 1 个谓词，2 条规则 | **7 个谓词，5 条规则** |
| 数据平衡 | 完全均衡 | 正例率 0.5%～81%，**极端不平衡** |
| 训练方式 | 纯 LTN 损失 | **混合损失**（LTN 30% + 加权 BCE 70%） |
| 坍塌问题 | 不存在 | 混合损失**完全解决**预测坍塌 |
| 评估方式 | 固定阈值 0.5 | **Youden 动态阈值**，适配各谓词 |
| 规则满足度 | 未量化 | **ltn_sat = 0.8712**，量化验证 |

---

## 🎯 核心实验结果

### 分类性能（测试集 913 条公理）

| 谓词 | AUC | F1 | 说明 |
|------|-----|----|------|
| empty | **1.000** ⭐ | 1.000 | 完美——空集符号特征唯一 |
| setrel | **1.000** ⭐ | 0.998 | 完美——集合关系符号明确 |
| functor | **0.992** ⭐ | 0.979 | 优秀——函子符号覆盖广 |
| manysym | **0.977** ⭐ | 0.928 | 优秀——符号密度有效编码 |
| commut | 0.832 | 0.076 | 排序能力好，极端不平衡（测试集仅 7 个正例）|
| theorem | 0.710 | 0.611 | 中等——缺乏独特符号特征 |
| definition | 0.584 | 0.221 | 受限——与 theorem 特征重叠 |

### 逻辑规则满足度（测试集）

| 规则 | 逻辑公式 | 满足度 |
|------|----------|--------|
| R1 | ∀x  Empty(x) ⇒ ¬Theorem(x) | 0.9170 |
| R2 | ∀x  SetRel(x) ⇒ ¬Definition(x) | 0.8902 |
| R3 | ∀x  Commut(x) ⇒ Functor(x) | **0.9933** ⭐ |
| R4 | ∀x  ManySym(x) ⇒ Theorem(x) | 0.6135 |
| R5 | ∀x  Definition(x) ⇒ ¬Theorem(x) | 0.9420 |
| **总体 ltn_sat** | — | **0.8712** ✅ |

训练损失从 **0.7009 → 0.1051**（降幅 85%），全程无预测坍塌，无 NaN。

---

## 📁 仓库结构

```
LTN-CNF-Axiom-Classification/
│
├── original_paper/                      # 原论文相关材料
│   ├── LTNtorch（原论文）.pdf            # LTNtorch 原论文
│   ├── LTNtorch_Presentation.pptx       # 原论文讲解 PPT
│   ├── 典型例子-binary_classification.ipynb  # 原论文官方示例 Notebook
│   ├── SBR.pdf                          # 相关论文 SBR
│   ├── SBR_Presentation.pptx            # SBR 论文讲解 PPT
│   └── README.md
│
├── my_experiment/                       # 本实验材料
│   ├── ltn_cnf.ipynb                    # 实验 Notebook（含完整输出）
│   ├── 最新LTN_CNF_讲解PPT.pptx         # 实验讲解 PPT
│   ├── LTN_CNF_实验报告.docx            # 完整实验报告
│   └── README.md
│
├── src/
│   └── ltn_cnf_v4.py                    # 实验源代码
│
├── data/
│   ├── node_dict.pkl                    # 符号表（2,685 个符号）
│   ├── statements                       # 公理文本（4,564 条）
│   └── README.md
│
├── requirements.txt                     # Python 依赖
├── .gitignore
├──  AI交互记录（节选整理典型提问）.md       # 整理后的典型AI交互记录
├── LICENSE
└── README.md
```

---

## 🚀 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/Xxxx37/LTN-CNF-Axiom-Classification.git
cd LTN-CNF-Axiom-Classification
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 运行实验

```bash
# 方式一：直接运行脚本
python src/ltn_cnf_v4.py

# 方式二：打开 Notebook 逐步运行（推荐，含完整输出）
jupyter notebook my_experiment/ltn_cnf.ipynb
```

---

## 🧠 方法说明

### 混合损失函数

本实验最核心的改进，解决纯 LTN 训练中的**预测坍塌（Prediction Collapse）**问题：

```
L(θ) = α × L_LTN  +  (1−α) × L_BCE
     = 0.3 × (1 − SatAgg)  +  0.7 × 加权BCE
```

正例权重动态计算：`w_k = min(neg_k / pos_k, 30)`，适配极端不平衡分布。

### 谓词神经网络

7 个谓词各自对应一个独立的 3 层 MLP，总参数量 **775,047**：

```
Linear(300→256) → LayerNorm → GELU → Dropout(0.15)
Linear(256→128) → LayerNorm → GELU → Dropout(0.10)
Linear(128→1)   → Sigmoid   → safe(ε=1e-7)
```

### 特征工程

将每条公理的符号集合转为 **300 维 TF-IDF 特征向量**（L2 归一化），  
相比纯词袋（BOW）能有效压制高频通用符号，突出区分性特征。

---

## 📚 参考文献

```bibtex
@article{carraro2024ltntorch,
  title   = {LTNtorch: PyTorch Implementation of Logic Tensor Networks},
  author  = {Carraro, Tommaso and Serafini, Luciano and Aiolli, Fabio},
  journal = {Journal of Machine Learning Research},
  volume  = {25},
  year    = {2024},
  note    = {arXiv:2409.16045}
}

@article{badreddine2022ltn,
  title   = {Logic Tensor Networks},
  author  = {Badreddine, Samy and d'Avila Garcez, Artur and
             Serafini, Luciano and Spranger, Michael},
  journal = {Artificial Intelligence},
  volume  = {303},
  pages   = {103649},
  year    = {2022}
}
```

---

## 📄 License

MIT License © 2025
