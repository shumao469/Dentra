# 预训练模型与本机部署审查

审查日期：2026-10-20。目标是提供待医生精修的初稿，不是建立D/F诊断真值或评价3D诊断性能。没有训练，也没有搜索阈值、模型组合或后处理参数。

| 候选 | 公开输出与权重 | 许可记录 | 本轮决定 |
|---|---|---|---|
| DentalSegmentator v100 | 上颅骨含上颌、下颌、上下牙群、下颌管；公开权重229,747,861字节 | 官方Slicer源码Apache 2.0；Zenodo权重记录CC BY 4.0 | 首个本地初稿后端，冻结现有fold 0权重；不输出逐牙FDI、精细牙槽骨定义或D/F诊断 |
| ToothSeg | 逐牙实例与编号路线；双分支；公开ToothSeg.zip约920MB | 官方代码Apache 2.0；Zenodo记录CC BY 4.0 | v0.3已下载与接入；采用独立冻结的保守组装，详见TOOTHSEG_AUDIT.md |

来源：[DentalSegmentator官方代码](https://github.com/gaudot/SlicerDentalSegmentator)、[DentalSegmentator权重](https://zenodo.org/records/10829675)、[ToothSeg官方代码](https://github.com/MIC-DKFZ/ToothSeg)、[ToothSeg权重](https://zenodo.org/records/14893540)。许可字段分别从代码许可文件和Zenodo API元数据读取，不能互相替代。使用与发表时保留原模型及nnU-Net引用；实际使用范围按所对应许可条款执行。

DentalSegmentator的类别是较大解剖结构。整个上颌/颅骨掩膜不等于项目所需的精细牙槽骨边界，上下牙群也不是逐牙实例。本轮保留原始语义，并拦截把牙群直接确认成单颗FDI或把整颌骨直接确认成牙槽骨的操作。它可以减少初始勾画工作，是否适合薄骨板与缺损精修需医生实际确认。

## 冻结与校验

选择的文件是Zenodo记录10829675的 `Dataset112_DentalSegmentator_v100.zip`。MD5为 `b71cd5230168d28a4f71b078265b76be`，本机计算SHA-256为 `bc5510cc93bc2100ab1faccb63512e09c1ca326c738b0a9939c074d82b38a4ac`。大小、MD5、SHA-256和ZIP CRC全部检查后才解压有效权重。

`config/model_manifest.json` 固定checkpoint、dataset和plans的SHA-256及公开文件内现有fold 0。这个fold 0是外部预训练模型自己的目录编号，与2.5D项目五折没有关系。不会对本地病例选择新权重或训练轮次。

推理使用固定nnU-Net 2.8.1、PyTorch 2.10.0 CUDA 12.8，读取旧版nnU-Net plans时框架可能输出兼容警告。本模块不重新规划、不重新训练、不改写plans文件。内部强度归一化、输入重采样和滑窗概率融合按原plans执行；这些属于模型推理。输出返回原始CBCT网格，之后没有掩膜平滑、闭运算、填洞、最大连通域筛选或小结构清除。模型自身仍可能漏掉薄骨板或误补缺损，因此不能把“没有后处理”当作缺损一定保真的证据。

## 本机条件

实测可见RTX 5090约32GB和RTX PRO 5000 Blackwell约48GB。固定使用GPU 1即RTX PRO 5000。WSL可见内存约31GiB，现有swap约64GiB；这些是观测值，本轮没有更改系统内存或swap配置。E盘剩余空间较少，模型权重和新依赖放在WSL Linux磁盘，影像任务默认写D盘工作目录。

Windows Slicer 5.10.0处理显示与精修，WSL `Ubuntu-22.04`运行隔离的 `dentra_v03` 环境。新增nnU-Net依赖只写这个venv；PyTorch等底层包从现有环境只读继承，所以该环境属于隔离覆盖层，不是完全独立的运行时副本。原2.5D环境没有新增nnU-Net或改动其timm/PyTorch版本。没有在Slicer内部安装PyTorch。

部署检查会核对权重指纹、nnU-Net版本、GPU可用性和空闲显存。没有满足条件时停止，不自动改分辨率、改patch、降到CPU或搜索其他参数。具体运行证据见 `validation/`；单个本地影像试跑只证明工程可运行，不构成内部或外部性能验证。
