# 单细胞 RNA-seq 的批次效应校正方法哪类更可靠

## 有哪些针对单细胞RNA-seq批次效应校正方法的系统性基准评测研究，比较了不同方法的性能？

[PMC13221981 · Abstract]指出批次效应校正方法的比较性评测受到的关注有限，本文系统研究了批次效应去除常用评估指标的行为和敏感性差异。[PMC13221981 · Abstract]该研究汇编并系统比较了多个评估指标，使用人工稀释序列方法生成含受控噪声的数据集来定量评估各指标区分噪声水平的能力。[PMC13221981 · 2 Methods]本研究使用了四个真实RNA-seq数据集，其中两个是单细胞RNA-seq数据集。[PMC13221981 · 4 Discussion]引用的近期工作（Rautenstrauch and Ohler 2025）对单细胞整合基准评测进行了研究，指出基于Silhouette的指标可能具有误导性，并建议混合使用局部和全局距离指标。

## 基于深度学习的批次校正方法（如scVI、Harmony等）与基于线性模型/回归的方法相比，在保留生物学信号方面表现如何？

深度学习类方法（如SCVI、LIGER）在测试中常大幅改动数据，产生可测量的伪影，表现较差 [PMC12315870 · Abstract]。这些方法引入了显著改变，破坏了局部邻域结构、聚类和差异表达结果 [PMC12315870 · Discussion]。相比之下，线性方法Combat与Harmony引入的伪影最少，其中Harmony是唯一在批次无局部偏差时减弱校正的方法，这解释了它为何能抵抗过度校正、保留生物学信号 [PMC12315870 · Discussion]。仅一篇文献报告，Harmony在所有测试中表现一致良好，而深度学习类方法在该研究中表现不佳 [PMC12315870 · Abstract]。

## 批次校正方法在保留真实细胞类型/稀有细胞群方面是否存在过度校正的风险，哪些方法更稳健？

KL正则化与对抗学习方法存在明确的过度校正风险，因为这类方法在定义上不区分生物学信息与批次信息，会同时移除两者 [PMC12577435 · Results]。增加KL正则化强度会提高批次校正效果，但降低生物学信息保留，进一步说明过度校正随校正强度增加而加剧 [PMC12577435 · Results]。依赖KL正则化和对抗学习的方法在增强批次校正时难以保留足够的生物学信息 [PMC12577435 · Discussion and outlook]。相比之下，结合VampPrior和循环一致性损失的sysVI模型在批次校正和生物学保留两方面均表现良好，被作者视为更稳健的方法 [PMC12577435 · Discussion and outlook]。作者建议方法开发社区从对抗学习等批次分布匹配技术转向基于循环一致性的方法，并用VampPrior替代标准高斯先验，以更稳健地保留生物学信息 [PMC12577435 · Discussion and outlook]。在标签存在缺陷的场景下，仅一篇文献报告scANVI和ssSTACAS保持稳定表现，但两者均未超过无监督方法scCRAFT，且scANVI相对scVI的优势很小 [PMC13020996 · 3 Conclusions]。同一篇文献报告，即使是适度的结构化标签缺陷也会使scDREAMER、ItClust和scGEN的表现跌至最强无监督方法scCRAFT及常用基线之下，说明这些半监督方法在标签不完美时稳健性差 [PMC13020996 · 3 Conclusions]。当标注质量不确定时，scCRAFT被报告为最可靠的默认整合方法，而scANVI仅在需要相对scVI的适度提升或主要目标是标签迁移时才值得考虑 [PMC13020996 · 3 Conclusions]。

## 不同批次校正方法在数据整合（data integration）任务中的基准比较，特别是基于互近邻（MNN）类方法的表现

在数据整合基准中，基于互近邻的方法表现并不一致。MNN Correct在恢复差异表达基因方面表现最好之一，与ComBat、ZINB-WaVE和scMerge并列 [PMC6964114 · Discussion]。然而，在整体批次校正效果排名中，Harmony、LIGER和Seurat 3被推荐为最佳方法，而MNN Correct未进入前列 [PMC6964114 · Abstract]。Seurat 3本身也使用互近邻锚点，但结合了典型相关分析，属于整体表现最好的方法之一 [PMC6964114 · Methods and materials]。另一项基准报告Scanorama在复杂整合任务上表现良好，尤其在嵌入空间上 [PMC8748196 · Discussion]。仅一篇文献报告fastMNN基于降维空间中的互近邻方法，但未给出其与其他方法的直接排名比较 [PMC6964114 · Methods and materials]。

## 引用文献

- `PMC12315870` Batch correction methods used in single-cell RNA sequencing analyses are often poorly calibrated
- `PMC12577435` Integrating single-cell RNA-seq datasets with substantial batch effects
- `PMC13020996` A benchmark of semi-supervised scRNA-seq integration methods in real-world scenarios
- `PMC13221981` Development and comparison of evaluation metrics for batch correction reveals performance differences
- `PMC6964114` A benchmark of batch-effect correction methods for single-cell RNA sequencing data
- `PMC8748196` Benchmarking atlas-level data integration in single-cell genomics