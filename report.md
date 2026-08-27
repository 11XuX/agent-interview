# 单细胞 RNA-seq 的批次效应校正方法哪类更可靠

## 有哪些针对单细胞RNA-seq批次效应校正方法的系统性基准评测研究，比较了不同方法的性能？

针对单细胞RNA-seq批次效应校正方法的系统性基准评测研究数量有限，其比较评估受到的关注较少 [PMC13221981 · Abstract]。一项研究将scBCN与八种方法（fastMNN、Harmony、Liger、Scanorama、scANVI、scMC、scVI、Seurat V4）进行了基准评测，并使用两个模拟数据集评估性能，其中数据集2来自一项综合性基准评测研究 [PMC12459263 · Results]。另一项研究系统比较了多个批次效应去除评估指标，使用人工稀释系列方法生成含受控批次效应噪声的数据集，并纳入四个真实RNA-seq数据集，其中两个为单细胞RNA-seq数据集 [PMC13221981 · Abstract][PMC13221981 · 2 Methods]。该研究还引用了近期工作（Rautenstrauch and Ohler 2025），指出基于Silhouette的指标在单细胞整合基准评测中可能具有误导性，并提出BRAS作为替代指标 [PMC13221981 · 4 Discussion]。

## 基于深度学习的批次校正方法（如scVI、Harmony等）与经典线性方法（如ComBat、MNN、Seurat CCA）相比，在保留生物学信号方面表现如何？

在PMC12315870的测试中，深度学习类方法SCVI表现差，常大幅改动数据，而经典方法ComBat、MNN、Seurat也引入可检测的伪影，只有Harmony在所有测试中表现良好 [PMC12315870 · Abstract]。同一研究的讨论部分报告ComBat和Harmony表现最好、引入伪影最少，Seurat引入较易识别的伪影但保留整体结构，而ComBat-seq、MNN、SCVI、LIGER、BBKNN引入显著改变，破坏局部邻域结构、聚类和差异表达结果 [PMC12315870 · Discussion]。然而，PMC12459263在模拟数据集1中报告fastMNN、Scanorama、scANVI、scVI和scBCN能成功整合批次并保留细胞类型边界，而Harmony、Liger、scMC、Seurat V4错误连接不同细胞类型或未能混合批次 [PMC12459263 · Results]。仅一篇文献报告scBCN在ARI和NMI上取得满分，并在ASW_celltype和iLISI指标上优于所有其他方法，表明其在生物学变异保留和批次校正方面表现更优 [PMC12459263 · Results]。PMC12315870讨论部分还指出ComBat虽在该测试中表现好，但在去除批次效应方面被更现代的方法超越，且作者只推荐Harmony用于一般用途 [PMC12315870 · Discussion]。

## 批次校正方法在保留真实细胞类型/稀有细胞群方面是否存在过度校正的风险，哪些方法更稳健？

现有整合方法中依赖显式批次校正的策略常导致过度校正并丢失有生物学意义的变异 [PMC13128840 · Discussion]。相比之下，scDecorr 不采用显式批次校正，而是通过域特定批归一化隐式混合批次，从而避免过度校正并保留域特异的生物学结构 [PMC13128840 · Discussion]。仅一篇文献报告，scDecorr 在形成共享细胞类型聚类的同时保留了批次特异和稀有细胞群 [PMC13128840 · Discussion]。在批次重叠有限或强不对称的困难场景中，scDecorr 避免无关细胞状态的激进混合并最小化过度校正 [PMC13128840 · Discussion]。该文献还报告 scDecorr 的高过度校正得分和竞争性批次混合性能表明其在技术校正与生物学保真之间取得良好平衡 [PMC13128840 · Discussion]。

## 不同批次校正方法在数据整合（data integration）任务中的定量评估指标（如kBET、LISI、ASW等）结果如何？

FedscGen在Human Pancreas数据集上的NMI、GC、ILF1、ASW_C、kBET、EBM等指标与scGen相当[PMC12285155 · Abstract]。FedscGen与scGen在批次混合和细胞类型区分方面结果相似，并在八个数据集和十个评估指标上与集中式方法性能相当[PMC12285155 · Discussion][PMC12285155 · Conclusions]。然而，Tran等人的基准测试显示没有任何单一方法在所有数据集和指标上始终优于其他方法[PMC12285155 · Discussion]。在指标方向上，RBET和kBET值越小表示批次校正效果越好，而LISI值越大表示效果越好[PMC11954866 · Results]。RBET在高斯模拟中与kBET、LISI表现相当，但在模拟基因表达数据中检测功效优于LISI，而kBET失去功效[PMC11954866 · Results]。仅一篇文献报告RBET在批次效应检测功效、I类错误控制、过校正敏感性、计算效率和鲁棒性方面均优于LISI和kBET[PMC11954866 · Discussion]。使用不同指标可能得出相反结论，例如曾被高度评价的Harmony在后续研究中被建议不要用于复杂数据集[PMC11954866 · Discussion]。

## 引用文献

- `2512.09259v1` MoDaH achieves rate optimal batch correction
- `PMC11954866` Reference-informed evaluation of batch correction for single-cell omics data with overcorrection awareness
- `PMC12285155` FedscGen: privacy-preserving federated batch effect correction of single-cell RNA sequencing data
- `PMC12315870` Batch correction methods used in single-cell RNA sequencing analyses are often poorly calibrated
- `PMC12459263` scBCN: deep learning-based batch correction network for integration of heterogeneous single-cell data
- `PMC13128840` scDecorr: feature decorrelation based representation learning enables self-supervised alignment of multiple single-cell experiments
- `PMC13221981` Development and comparison of evaluation metrics for batch correction reveals performance differences