# Dentra 逐牙初稿与 Slicer 精修操作指南

版本 v0.3 · 2026-10-20 · Windows Slicer与WSL本地推理

## 1 使用范围与快速开始

本指南面向医生与本地工程维护人员，说明如何用固定预训练模型生成CBCT三维分割初稿，再进入Slicer精修、独立复核和版本保存。已有精标仍可直接打开精修。当前工具不训练模型，不接入2.5D推理，不使用G04。

### 本轮可以完成的工作

Dentra v0.3新增逐牙模式：ToothSeg双分支固定权重输出牙体实例，采用保留前景的Dentra组装并提供FDI建议。先在“初稿模型”选择模式，再检查部署和生成。五类结构与逐牙结果分开保存，不互相覆盖。

DentalSegmentator v100生成五类结构初稿：上颅骨含上颌、下颌骨、上牙群、下牙群和下颌管。随后在工作副本上使用画笔、擦除等工具精修，保留模型初稿与医生改动的来源区别。

这些类别不等于逐牙FDI实例或精细牙槽骨标注。模型不输出D/F诊断。没有自动平滑或填洞，仍不能保证模型预测保留每处薄骨板或真实缺损；薄骨板、开窗和开裂必须分别看原图复核。

### 本机已部署时的最短流程

1. 解压整个交付目录到固定位置，双击Start_Dentra.cmd。
2. 在Slicer中载入独立3D标注流程的CBCT，选择匿名病例ID和已加载影像。
3. 点击“检查本机模型与GPU”；通过后确认输入范围，点击“生成初稿并进入精修”。
4. 初稿导入后，核对影像对应关系，逐段确认模型类别；在副本上精修。
5. 分别复核薄骨板、开窗和开裂，并逐一处理疑点标记。
6. 运行质控，另存新版本；下次使用同一CBCT、初稿源文件和版本目录恢复。

### 阅读导航

首次部署见第2和第3节；医生日常操作见第4至第8节；版本恢复与故障处理见第9和第10节。第11节给出测试边界和验收清单，第12和第13节说明逐牙编号与模型限制，第14节供工程人员接管。

## 2 本机环境与模型类别

### 部署位置

| 组件 | 本机配置 |
| --- | --- |
| 显示与精修 | Windows 3D Slicer 5.10.0 |
| 推理环境 | WSL Ubuntu-22.04；隔离venv dentra_v03 |
| GPU | GPU 1 NVIDIA RTX PRO 5000 Blackwell，约48GB显存 |
| 推理软件 | nnU-Net 2.8.1；PyTorch 2.10.0 CUDA 12.8 |
| 权重目录 | /home/shu/models/cbct_draft/DentalSegmentator_v100 |
| 标注数据 | E:\3D；仅本地读取，不随代码包分发 |

WSL可见内存约31GiB，现有swap约64GiB。本轮未更改系统配置。E盘余量较少，权重和依赖放在WSL磁盘，任务目录默认写入D盘工作区。环境检查要求GPU可用且空闲显存至少12GiB；不满足时先查占用，不改模型分辨率或patch来强行运行。

### 初稿类别的含义

| 标签 | 界面类别 | 确认边界 |
| --- | --- | --- |
| 1 | upper_skull_maxilla | 上颅骨及上颌范围；不是精细牙槽骨真值 |
| 2 | mandible | 下颌骨整体类别 |
| 3 | upper_teeth_group | 上牙群；无逐牙编号 |
| 4 | lower_teeth_group | 下牙群；无逐牙编号 |
| 5 | mandibular_canal | 下颌管模型类别 |

不在视野内的类别可能没有前景。工具不为缺失类别造出掩膜。临床牙位、组织边界与D/F定义仍由医生确认。

逐牙模式另需ToothSeg公开权重，约920MB；模型根目录写在config/local_backend.json的toothseg_model_root。本机使用GPU 1，要求至少36GiB空闲显存、输入不超过1.25亿体素。模型内部按checkpoint计划归一化与重采样，最终回到原始CBCT格点；原图和现有精标不改写。

## 3 首次部署与文件校验

本轮已在当前机器完成部署，日常使用可直接进入下一节。迁到另一台电脑、移动WSL环境或更换底层PyTorch后，应重新检查部署，不能沿用本机测试结论。

### 一次性安装步骤

1. 保持整个交付目录结构完整，不要只复制CBCTSegReview.py。
2. 检查config/local_backend.json中的WSL发行版、隔离Python和模型目录。本机值已填写。
3. 在WSL Bash终端进入解压目录；用已有的科研Python执行tools/install_backend.py。不要在Slicer Python控制台执行安装脚本。
4. 脚本下载公开权重、校验大小及MD5/SHA-256、检查ZIP CRC，只解压清单内的有效模型文件；在独立venv安装新增依赖，最后运行预检。

```bash
/home/shu/miniconda/envs/cxr_torch5090/bin/python \
  tools/install_backend.py
```

该venv通过system-site-packages只读继承现有PyTorch等基础包，新增nnU-Net依赖写入venv自身。这是隔离覆盖层，不是完全独立复制的运行时；原2.5D环境没有安装nnU-Net或改变timm/PyTorch版本。

### 权重校验依据

公开文件为Dataset112_DentalSegmentator_v100.zip，大小229,747,861字节。官方MD5：b71cd5230168d28a4f71b078265b76be。SHA-256及每个checkpoint/plans/dataset文件的指纹保存在config/model_manifest.json。

下载中断会保留.partial文件并尝试续传；校验未通过不会加载模型。代码ZIP不包含模型权重或患者影像。DentalSegmentator权重记录采用CC BY 4.0，相关代码采用Apache 2.0；保留原模型与nnU-Net文献引用。[1–3]

### 启动入口

双击Start_Dentra.cmd可临时加载模块，不改变永久模块路径。若Slicer安装位置不同，在PowerShell中运行Launch_Slicer_Review.ps1，并用-SlicerPath指定Slicer.exe。CMake扩展发布路径尚未测试，本机验证的是直接模块加载。

### 从GitHub克隆后的配置

GitHub仓库不包含本机配置、患者数据或模型权重。将config/local_backend.example.json复制为local_backend.json并修改USER、WSL发行版和GPU配置适配前先审查冻结协议。若当前机器已部署，直接保留本机local_backend.json。

```bash
python tools/install_backend.py --provider toothseg
python tools/install_backend.py --provider dental
```

安装命令在兼容CUDA PyTorch的WSL科研Python下运行，安装器在独立venv中安装新增依赖；不要使用Slicer Python。迁移到显存不足的机器时先停止，不自动修改patch、spacing或降低模型。

## 4 载入影像并检查部署

### 在Slicer载入正确的CBCT

1. 打开Slicer后，在DICOM模块导入病例所在目录，选中正确序列并载入；已有NRRD/NIfTI也可按Slicer常规方式载入。
2. 进入“CBCT Research → Dentra”。输入匿名病例ID，如E3D_CASE_001。
3. 在“已加载CBCT / Image”中选择目标体积，核对左右、上下颌、视野和空间位置。
4. 如果准备精修已有标注，选择原始.seg.nrrd并点“载入并创建副本”；如果准备生成初稿，使用下方“1b 预训练初稿”。

匿名ID仅允许字母、数字、下划线和短横线，不能填姓名、空格或路径。G04标记会被拒绝；来自DICOM的患者身份和队列归属不能仅靠文件名自动判定，仍需操作者核对。不要把G04或2.5D测试数据换名后放入本流程。

### 检查本机模型与GPU

1. 点击“检查本机模型与GPU / Check deployment”。
2. 日志应出现READY，界面提示Deployment check passed。检查会核对模型文件指纹、nnU-Net版本、GPU可用性和空闲显存。
3. 失败时先看错误信息，按第10节处理；不启动其他模型、不降低输入分辨率或修改后处理参数。

### 选择任务目录

“初稿任务目录”应使用本地Windows盘符路径，例如D:\CBCT_Draft_Jobs。不要使用网络盘或UNC共享路径。每次生成都会新建唯一任务子目录，不覆盖已有任务。目录内含完整CBCT和生成掩膜，应留在本地受控数据位置，不提交Git、不放进公开交付ZIP。

已有精修会话或生成任务打开时，先保存并关闭会话或等待任务结束，再开始另一个任务。一个模块实例只运行一个后端任务。

## 5 生成半自动初稿

### 启动和观察进度

1. 确认输入来自独立3D标注流程，勾选“不含G04或2.5D测试病例”的确认框。
2. 检查病例ID、选中影像和任务目录，点击“生成初稿并进入精修”。
3. 工具先在本地导出选中体积，不改变原始体积的保存目标，再通过WSL调用固定模型。
4. 日志显示预处理、网络前向调用数和完成状态。前向次数不是完成百分比，不应据此估计所有病例耗时。
5. 成功后自动导入Slicer并建立工作副本。患者—影像确认和所有解剖复核仍是待确认状态。

DentalSegmentator使用其公开fold 0；ToothSeg双分支使用其公开fold 5，两者均与2.5D项目固定五折无关。推理使用原plans中的归一化与重采样，并在原始CBCT网格输出结果。滑窗的Gaussian权重用于融合预测，不是对最终二值掩膜做形态学平滑。输出之后不平滑、不闭运算、不填洞、不删除小连通域。

### 本机试跑的实际范围

DentalSegmentator在一个本地241×685×685体积上的历史试跑耗时约161秒，并通过Slicer导入、原网格检查与草稿版本保存。该时间仅描述这一次试跑，不是其他病例或其他机器的耗时承诺。没有计算Dice、D/F准确率或外部评价成绩。

### 取消或中途关闭

点击“取消初稿任务”后，后端在阶段或网络前向边界检查取消标记并退出，界面可能需要等待。取消后不会自动导入结果。失败与部分目录保留供检查；重新尝试时建立新任务，不将.partial_result当成完成结果。

若推理已完成但Slicer关闭，可重新载入同一CBCT，点击“导入已完成初稿任务”，选择含job.json和result子目录的任务根目录。工具会核对影像像素与几何、输入/输出哈希及模型来源。

## 6 确认类别并安全精修

### 先确认来源与语义

1. 初稿导入后，先核对影像与掩膜物理位置，再勾选患者—影像绑定确认；未确认时编辑保持只读。
2. 在段选择框逐段查看类别、原始名称和像素值。五类结构初稿保留jaw_bone、teeth_group和nerve_canal语义；ToothSeg逐牙初稿选择tooth并核对FDI建议。
3. 牙群不要填写FDI；整颌骨不要直接改认成alveolar_bone。需要精细牙槽骨范围时，先另定标注协议；逐牙模型只能提供待精修的牙体建议。
4. 已有人工逐牙精标可选择tooth并明确填写FDI；名称和像素标签值不等价，禁止按像素值自动编号。

### 精修工具

| 工具 | 用途 | 操作后检查 |
| --- | --- | --- |
| Paint / Draw | 局部补画、沿原图勾边 | 新增骨体素是否误补真实缺损 |
| Erase | 移除多余体素 | 是否误删薄骨板或窄连接 |
| Scissors | 手工裁去明确多余区域 | 范围是否波及目标牙或缺损边缘 |
| Undo | 撤销最近操作 | 返回原图再次确认修改依据 |

默认不覆盖其他segment。勾选“对照原始轮廓”会隐藏工作副本并暂停编辑，取消勾选后回到副本。模型初稿本身是对照基线，不是人工真值。

可选3D表面显示使用平滑系数0且不改变二值掩膜；判断薄骨板或缺损仍以原始切片及相邻切片为依据。不要切换到其他通用工具批量平滑或填洞，再把结果当成本模块未后处理的输出。外部改动可能被质控拦截，但本工具不对Slicer所有入口施加系统级只读锁。

## 7 薄骨板 开窗和开裂的独立复核

先完成掩膜、映射和疑点位置调整，再记录最终复核。任何相关内容变化都会使旧复核签名过期。没有后处理不等于预测不会漏骨或误补缺损；这些区域必须由医生从原图独立判断。

| 复核项 | 重点查看 | 记录内容 |
| --- | --- | --- |
| 薄骨板 thin_plate | 原始及相邻切片的局部骨边界；必要时从另一方向复查 | 部位、可判定性、增删依据；模糊处保留不确定 |
| 骨开窗 fenestration | 疑似窗孔周围连续边界；新增体素是否填窗 | 牙位或区域、边界、原图证据和保留情况 |
| 骨开裂 dehiscence | 缺损与牙槽嵴边缘的连续关系及相邻切片 | 范围、边缘与修改依据；不要由线段名称自动定类 |

### 复核结论

“已复核”表示医生完成当前内容的检查；“不适用”必须说明原因；“仍不确定”保留待处理状态。每类整体项目都要有记录，每个已标记疑点也要单独复核，不能用整体通过代替疑点检查。

复核时填写医生匿名ID、结论及依据，包括查看部位、切片或坐标和修改理由。记录自动附时间与版本签名。解剖定义、可判定规则和正式双医生真值流程仍由项目医生确认，本模块没有引入新的D/F诊断阈值。

### 不可判定时的处理

若影像不足以区分薄骨、缺损或伪影，保留“仍不确定”，提交另一位专家复查。不要为获得完成状态而把不确定改成阴性或填平孔洞。缺损复核不依赖历史D线中点、L轴或P平面自动生成真值。

三维点位只帮助导航，不覆盖整个病变体积，也不自动发现所有风险区域；医生仍需检查相关完整区域。

## 8 疑点定位与质控结果

### 放置与细调疑点

1. 选择对应segment与疑点类型，填写“位置/原因”，例如“11唇侧根中部，窗孔边缘待核对”。
2. 点击图钉，在切片放一个点，再点“将放置点加入复核清单”。多放了点时先清除临时放置点。
3. 双击清单项回到该位置。需要移动时，选中疑点、在新位置放一个临时点、更新说明，再点“移动选中疑点”。
4. 明确误标可选中删除。新增、移动或删除点后，重做相关最终复核；点位变化会更新签名。
5. 在复核项下拉框中逐项记录三个整体类别及各疑点的结论。

### 运行质控后怎样处理

| 提示 | 含义与处理 |
| --- | --- |
| added_voxels / removed_voxels | 相对基线的增删量；骨新增时重点检查真实缺损是否被补上 |
| outside_reference_volume | 掩膜超出参考影像；先核对数据与几何，不直接确认完成 |
| different_sampling | 采样与CT不同；保持原网格，先检查物理位置与细节 |
| mask_grid_changed | 精修中标签格点改变；本版不通过插值强行比较 |
| review_pending_or_stale | 未复核、仍不确定或内容修改后旧记录已过期 |
| check_volume_boundary | 结构接触视野边缘；结合FOV检查，不能据此自动判缺损 |

IN_PROGRESS表示仍有待处理项目，可保存草稿。REVIEW_COMPLETE只表示本工具所需工程检查与复核记录齐备；training_admission和clinical_validation仍为false。模型来源不会因为精修完成就自动变成人工真值。

## 9 保存版本与下次恢复

### 保存一次完整版本

1. 运行质控，阅读仍未解决的项目。可以先保存IN_PROGRESS草稿。
2. 在“导出根目录”选择本地受控位置，点击“另存新版本”。
3. 每次生成新的匿名ID、UTC时间和随机版本号目录。保存后会重新读取分割，核对原生二值掩膜指纹。
4. 关闭会话前必须先导出尚未保存的掩膜、映射与复核更改。恢复其他版本前也应先保存当前改动。

| 版本内文件 | 用途 |
| --- | --- |
| refined.seg.nrrd | 精修掩膜；模型来源标签随segment保留 |
| critical_regions.mrk.json | 疑点类别、点位及说明 |
| review.json | 源文件指纹、影像签名、映射、体素差分、复核人与时间、模型来源 |
| SHA256SUMS.txt | 上述文件的SHA-256校验 |

### 恢复已有精标的精修版本

先载入同一CBCT，用原始.seg.nrrd建立会话，再点“恢复已导出版本”，选择版本目录；重新确认患者—影像绑定。不要用精修后的文件替代原始基线，否则差分含义会改变。

### 恢复模型初稿的精修版本

先载入同一CBCT，点“导入已完成初稿任务”选择原任务目录，重建同一模型基线；再点“恢复已导出版本”选择精修版本目录。也可用任务内slicer_source/model_draft.seg.nrrd作为原始精标入口，来源标签仍应保留。

版本目录不包含CBCT。恢复依赖原图与原始基线，不只是一个ZIP。SHA-256通过仅证明文件一致，不能替代患者身份或解剖复核。不要用Slicer通用Save入口覆盖原文件。

## 10 常见故障与恢复

| 现象 | 检查与处理 |
| --- | --- |
| Slicer.exe找不到 | 确认安装位置，用Launch_Slicer_Review.ps1的-SlicerPath参数指定 |
| 无法启动WSL后端 | 检查Ubuntu-22.04是否存在，以及config/local_backend.json的Python路径 |
| checkpoint或plans校验失败 | 停止推理，保留异常文件，按固定清单重新下载；不跳过校验 |
| GPU不可用或显存不足 | 检查其他任务及GPU编号；不要未经审查改变模型网格、patch或权重 |
| 旧plans格式警告 | 本机兼容载入已验证；不要执行重新规划或训练命令。若伴随异常，查FAILED.json |
| 日志暂时不变 | 预处理、回到原网格和压缩保存阶段可能耗时；先查看进程和资源，不重复点击生成 |
| 几何或影像签名不匹配 | 重新选对原图与原任务，核对坐标；不靠强行重采样或换病例ID绕过 |
| 复核完成后又变待处理 | 检查掩膜、名称、映射或点位是否变化；对新内容重新复核 |
| 模型少了某个类别 | 先确认是否超出FOV或预测遗漏；不自动补造掩膜，不将缺失当阴性 |
| 导出失败或.partial目录 | 读取EXPORT_FAILED.txt或FAILED.json；保留旧完整版本，新建任务/版本再处理 |

提问或报错时提供模块版本、匿名病例ID、所处步骤、错误日志和任务状态；不要把患者影像、姓名或DICOM标识发到公开工单。需要临床判读的薄骨板或缺损问题保留“不确定”，交医生复核。

## 11 验证证据与医生最小验收

### 本轮已验证的范围

当前工程证据包括核心规则回归、Slicer真实Qt精修流程、合成薄板与孔洞交换测试、Qt到WSL真实预检，以及一个E:\3D体积的真实预训练推理与Slicer草稿保存。详细计数和代码指纹见validation目录及VALIDATION.md。

试跑前后6份原始精标及该影像的241份DICOM内容指纹保持不变。没有用精标给模型打分，没有训练3D模型或比较D/F诊断成绩，也没有访问G04做调参或正式评价。

### 医生操作验收清单

1. 核对一个病例的患者、序列、左右和空间位置，确认可顺利启动与生成初稿。
2. 确认五类结构与逐牙模式使用各自正确语义，逐牙FDI建议需逐项确认或修正；两者均不提供精细牙槽骨真值。
3. 在副本做少量明确增删，检查撤销、基线对照与增删统计。
4. 分别看薄骨板、开窗和开裂，至少放置并移动一个疑点。
5. 确认修改后旧复核失效，能够填写依据并保留不可判定项。
6. 保存、关闭、恢复同一版本，确认影像、掩膜、点位和来源记录一致。

### 需要医生另行确定的规范

精细牙槽骨的目标边界、逐牙实例拆分规则、D/F解剖定义、不可判定项与双医生复核方案尚不能由预训练模型替代。当前复核工具可先试用，半自动初稿仍全部作为待确认内容。

English acceptance boundary: DentalSegmentator supplies five aggregate classes. The separate ToothSeg mode supplies tooth instances and unconfirmed FDI proposals. Neither mode supplies fine alveolar-bone truth or D/F diagnosis. Engineering tests and one local inference do not establish clinical accuracy. Every draft remains excluded from training until a separate admission process is approved.

## 12 逐牙实例与 FDI 确认操作

逐牙模式面向恒牙FDI 11至48。实例ID只是分割对象编号，模型语义类编号也不是FDI。FDI建议经固定dataset.json映射得到，并且与医生最终确认值分别保存。乳牙、混合牙列、种植体或异常牙位的适用性尚未验证。

1. 启动Start_Dentra.cmd，载入原始CBCT，选择匿名ID和正确影像。
2. 选择“逐牙实例与FDI建议 / ToothSeg”，点击检查部署。确认来源不含G04或2.5D测试病例后生成。
3. 逐段检查Instance编号、轮廓和FDI建议。PENDING和unknown都表示待医生确认。核对患者左右，不按屏幕左右直接推断FDI。
4. 在组织角色选择“牙”，FDI框填写或修正实际牙位，再点击确认当前段。不确定时保持未确认并在说明记录原因。
5. 查看重复FDI或无编号实例。不要为了消除提示而合并或删除碎片；先检查是否同牙断片、邻牙粘连、伪影或预测遗漏。
6. 用画笔、绘制、擦除和剪刀在工作副本精修。模型原始建议和原始前景文件始终保留。
7. 分别完成薄骨板、开窗、开裂及已放置疑点的人工复核，运行质控后导出新版本。仍不确定可导出IN_PROGRESS草稿。

### 重复编号和未知编号

同一FDI建议可以出现在多个独立实例上。模型建议不会导致自动合并，医生确认后的重复FDI会阻止REVIEW_COMPLETE。医生应根据原图修正错误牙位；若确为同牙多个碎片或漏掉新牙体，当前模块不支持改变segment数量，需记录待处理并由工程人员准备保留来源的拆分合并流程，不能靠改成其他组织角色绕过。

hard_vote_fraction只反映实例内离散语义投票比例，不是校准置信度、正确率或伪标签准入阈值。即使投票为1，也必须人工确认。对背景占优、票数并列或无核心片段，保留unknown或风险标记。

### 与骨结构一起复核

ToothSeg不输出牙槽骨。需要颌骨参考时可另建五类结构任务或加载已有医生骨标注；不得把牙实例当骨、把整个颌骨改为精细牙槽骨真值。当前自动导入每次只建立一种初稿会话，先导出并关闭，再打开另一任务。不同任务的原图必须人工确认一致。

## 13 ToothSeg 模型与保守组装限制

本轮固定ToothSeg官方源码提交b29d1017fa124f89645fa56f98649e3f3f43bdb0及Zenodo 14893540公开双分支权重。下载核对920019129字节、公开MD5、SHA256与ZIP CRC；加载前核对checkpoint、plans和dataset清单。权重单独下载，不包含在GitHub或工具ZIP中。

### 保留前景的组装规则

边界核心分支在原始CBCT格点输出后，以六连通核心作标记，只在预测前景内分配边界。没有核心的独立前景也保留为实例。软件断言实例前景与原始边界核心前景逐体素相等；Slicer导入时再次验证。小碎片、孔洞和重复编号不被自动删除、填补或合并。

这与官方完整后处理不同：官方流程可按体积删除小实例、重分配碎片、使用概率与牙弓约束修正编号。Dentra不用这些清理与搜索步骤，采用离散语义投票，因此不能引用论文精度作为本工具精度。两分支内部的归一化与重采样仍按冻结plans执行，不代表模型没有改变影像细节。

### 预训练权重仍有解剖局限

官方训练数据转换对牙体参考mask调用填洞。Dentra不运行该转换、不训练、不对本地mask填洞，但权重可能继承该标注习惯。保留模型预测前景不等于保证真实缺损完整。薄骨板、骨开窗和开裂始终需要独立人工查看原始影像。

### 可验证范围与仍未证明的内容

本机曾在同时展开33个原网格浮点通道时遇到高内存开销。现改为逐通道调用同一冻结重采样函数并逐步求argmax；16组合成数值对照的原网格标签完全一致。没有改权重、spacing、patch或镜像增强。失败任务保留，新任务另建；日志中的native_channel_export是正常输出阶段。

固定合成测试检查前景、孔洞、小碎片、重复和未知FDI、来源哈希、坐标及版本恢复。一份本地影像用于双分支执行和Slicer往返工程检查，结果以validation中的实际报告为准。未进行Dice、FDI准确率、D/F诊断成绩或独立外部临床验证；未使用G04调参或正式评价。

Archived custom-trainer compatibility is handled by constructing the exact saved plans architecture and strictly loading every tensor key. Checkpoint mirroring axes are retained. This verifies load compatibility, not numerical equivalence to the original ToothSeg software stack. No base research environment, checkpoint or 2.5D code is patched.

## 14 工程接管与参考资料

### 文件入口

| 文件 | 作用 |
| --- | --- |
| CBCTSegReview.py | Slicer界面、异步任务、精修与复核 |
| CBCTSegReviewLib/draft_bridge.py | 本地影像导出、WSL命令及来源校验后导入 |
| backend/infer.py | 固定预训练权重推理；没有训练入口 |
| config/draft_protocol.json | 冻结算法设置与复核约束 |
| config/model_manifest.json | 模型、plans与checkpoint校验清单 |
| config/local_backend.json | 本机WSL与环境路径 |
| tools/install_backend.py | 一次性部署及固定文件校验 |
| CODEX_START_PROMPT.txt | 后续会话接管约束 |

逐牙新增入口：backend/toothseg_infer.py执行双分支；CBCTSegReviewLib/toothseg.py执行保守组装与FDI提议；config/toothseg_protocol.json和toothseg_manifest.json分别锁定协议与权重。所有模型初稿保持待确认。

独立分支为feature/dentra-toothseg-fdi-v0.3。修改掩膜读写、来源校验、几何或复核逻辑后，重跑相应核心与Slicer测试；不能只沿用旧报告。代码包与Git bundle不含患者图像、精标、模型权重或任务目录。

### 英文状态词

READY：本机部署检查通过。MODEL_DRAFT：模型初稿。IN_PROGRESS：待完成。REVIEW_COMPLETE：本工具复核记录齐备。CANCELLED：取消。FAILED：未成功发布结果。任何状态均不自动代表临床有效或训练准入。

### 参考资料

[1] DentalSegmentator官方实现与许可：https://github.com/gaudot/SlicerDentalSegmentator

[2] 预训练权重与CC BY 4.0记录：https://zenodo.org/records/10829675

[3] nnU-Net官方代码与推理接口：https://github.com/MIC-DKFZ/nnUNet

[4] Slicer Segment Editor官方操作说明：https://slicer.readthedocs.io/en/latest/user_guide/modules/segmenteditor.html

[5] ToothSeg官方实现与权重：https://github.com/MIC-DKFZ/ToothSeg ；https://zenodo.org/records/14893540
