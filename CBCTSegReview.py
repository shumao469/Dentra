"""Slicer UI for doctor-guided refinement of existing binary segmentations."""
from pathlib import Path
import json

import ctk
import qt
import slicer
from slicer.ScriptedLoadableModule import ScriptedLoadableModule, ScriptedLoadableModuleWidget

from CBCTSegReviewLib import core, draft, toothseg
from CBCTSegReviewLib.slicer_adapter import ReviewSession
from CBCTSegReviewLib import draft_bridge


LABELS = {"thin_plate": "薄骨板 / Thin cortical plate", "fenestration": "骨开窗 / Fenestration",
          "dehiscence": "骨开裂 / Dehiscence"}


class CBCTSegReview(ScriptedLoadableModule):
    def __init__(self, parent):
        super().__init__(parent)
        parent.title = "Dentra"
        parent.categories = ["CBCT Research"]
        parent.dependencies = ["Segmentations", "SegmentEditor", "Markups"]
        parent.contributors = ["CBCT project"]
        parent.helpText = "已有精标精修与预训练模型初稿。原生网格保存，无平滑或填洞；模型初稿需医生独立复核。"
        parent.acknowledgementText = "Independent 3D annotation branch."


class CBCTSegReviewWidget(ScriptedLoadableModuleWidget):
    def setup(self):
        super().setup()
        self.session = None
        self.pickNode = None
        self.draftProcess = None
        self.draftJob = None
        self.pendingVolume = None
        self.layout.addWidget(qt.QLabel("Dentra v0.3 逐牙初稿与精修 / Draft and review\n编辑副本；不平滑或填洞；薄骨板与D/F逐项人工复核。"))
        load = self.section("1. 原始精标与影像 / Sources")
        self.caseEdit = qt.QLineEdit()
        self.caseEdit.placeholderText = "匿名病例ID，如 SAMPLE_A"
        load.addRow("匿名ID / Case", self.caseEdit)
        self.volumeBox = slicer.qMRMLNodeComboBox()
        self.volumeBox.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.volumeBox.addEnabled = False
        self.volumeBox.removeEnabled = False
        self.volumeBox.noneEnabled = True
        self.volumeBox.setMRMLScene(slicer.mrmlScene)
        load.addRow("已加载CBCT / Image", self.volumeBox)
        self.segEdit = qt.QLineEdit()
        load.addRow("原始 .seg.nrrd", self.segEdit)
        self.button(load, "选择原始精标… / Browse", self.browse)
        self.button(load, "载入并创建副本 / Start", self.start)
        self.button(load, "恢复已导出版本… / Resume", self.resume)
        self.binding = qt.QCheckBox("已人工核对患者、序列和物理位置对应 / Confirm image–mask binding")
        self.binding.toolTip = "几何一致不能证明患者相同。核对原图和标注来源后才能勾选。"
        load.addRow(self.binding)
        self.binding.connect("toggled(bool)", self.bind)
        self.button(load, "关闭会话（需先导出） / Close", self.closeSession)

        initial = self.section("1b. 预训练初稿 / Pretrained draft")
        initial.addRow(qt.QLabel("逐牙模式：ToothSeg双分支权重＋Dentra保守组装，FDI待人工确认。\n五类结构模式：DentalSegmentator。两种初稿分别保存；均不输出D/F诊断。"))
        self.providerBox = qt.QComboBox()
        self.providerBox.addItem('五类结构 / DentalSegmentator', draft.MODEL_ID)
        self.providerBox.addItem('逐牙实例与FDI建议 / ToothSeg', toothseg.MODEL_ID)
        initial.addRow('初稿模型 / Provider', self.providerBox)
        self.draftConsent = qt.QCheckBox("确认选中影像来自独立3D标注流程，不含G04或2.5D测试病例")
        initial.addRow(self.draftConsent)
        self.jobRootEdit = qt.QLineEdit(str(Path(__file__).resolve().parents[2] / 'work' / 'cbct_draft_jobs'))
        initial.addRow("初稿任务目录", self.jobRootEdit)
        self.button(initial, "选择任务目录… / Job directory", self.browseJobs)
        self.button(initial, "检查本机模型与GPU / Check deployment", self.checkBackend)
        self.button(initial, "生成初稿并进入精修 / Generate draft", self.generateDraft)
        self.button(initial, "取消初稿任务 / Cancel", self.cancelDraft)
        self.button(initial, "导入已完成初稿任务… / Import completed job", self.importCompletedJob)
        self.draftLog = qt.QPlainTextEdit()
        self.draftLog.readOnly = True
        self.draftLog.maximumHeight = 100
        initial.addRow(self.draftLog)

        mapping = self.section("2. 确认组织与FDI / Segment mapping")
        self.segmentBox = qt.QComboBox()
        mapping.addRow("原始Segment", self.segmentBox)
        self.segmentBox.connect("currentIndexChanged(int)", self.selectSegment)
        self.roleBox = qt.QComboBox()
        for text, value in (("待确认 / Pending", ""), ("牙 / Tooth", "tooth"), ("牙槽骨 / Alveolar bone", "alveolar_bone"),
                            ("颌骨模型类别 / Jaw bone", "jaw_bone"), ("牙群无FDI / Teeth group", "teeth_group"),
                            ("下颌管 / Mandibular canal", "nerve_canal"), ("其他 / Other", "other")):
            self.roleBox.addItem(text, value)
        mapping.addRow("组织角色", self.roleBox)
        self.fdiEdit = qt.QLineEdit()
        self.fdiEdit.placeholderText = "仅牙段填写，如11；禁止按像素值自动推断"
        mapping.addRow("FDI", self.fdiEdit)
        self.mappingNote = qt.QLineEdit()
        mapping.addRow("对应关系说明", self.mappingNote)
        self.button(mapping, "确认当前段对应关系 / Confirm mapping", self.confirmMapping)

        edit = self.section("3. 副本精修 / Manual editing")
        self.parameterNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentEditorNode", "CBCTReviewEditor")
        self.parameterNode.SetOverwriteMode(slicer.vtkMRMLSegmentEditorNode.OverwriteNone)
        self.editor = slicer.qMRMLSegmentEditorWidget()
        self.editor.setMRMLSegmentEditorNode(self.parameterNode)
        self.editor.setMRMLScene(slicer.mrmlScene)
        self.editor.setMaximumNumberOfUndoStates(20)
        self.editor.setUndoEnabled(True)
        self.editor.setEffectNameOrder(["Paint", "Draw", "Erase", "Scissors"])
        self.editor.unorderedEffectsVisible = False
        self.editor.setSegmentationNodeSelectorVisible(False)
        self.editor.setSourceVolumeNodeSelectorVisible(False)
        self.editor.setReadOnly(True)
        edit.addRow(self.editor)
        # Keep editing grids and segment identities fixed within this UI.
        for name in ("AddSegmentButton", "RemoveSegmentButton", "SpecifyGeometryButton"):
            child = slicer.util.findChildren(self.editor, name=name)
            for item in child:
                item.hide()
        self.baselineToggle = qt.QCheckBox("对照原始轮廓（隐藏工作副本） / Show baseline")
        self.baselineToggle.connect("toggled(bool)", self.showBaseline)
        edit.addRow(self.baselineToggle)
        self.surfaceToggle = qt.QCheckBox("显示3D表面（平滑系数0，仅显示） / Unsmoothed surface")
        self.surfaceToggle.connect("toggled(bool)", self.showSurface)
        edit.addRow(self.surfaceToggle)

        review = self.section("4. 薄骨板与D/F单独复核 / Critical regions")
        review.addRow(qt.QLabel("先标疑点，再完成复核。新增骨体素尤其要检查是否填平真实缺损。\n切片是判断依据；3D表面不能排除薄骨板或缺损。"))
        self.categoryBox = qt.QComboBox()
        for key in core.CATEGORIES:
            self.categoryBox.addItem(LABELS[key], key)
        review.addRow("疑点类型", self.categoryBox)
        self.regionNote = qt.QLineEdit()
        self.regionNote.placeholderText = "如：11唇侧根中1/3，骨窗边缘"
        review.addRow("位置/原因", self.regionNote)
        self.placeWidget = slicer.qSlicerMarkupsPlaceWidget()
        self.placeWidget.setMRMLScene(slicer.mrmlScene)
        self.placeWidget.buttonsVisible = False
        self.placeWidget.placeButton().show()
        review.addRow("点击图钉后在切片放点", self.placeWidget)
        self.button(review, "将放置点加入复核清单 / Add marker", self.addRegion)
        self.button(review, "清除临时放置点 / Clear placement", self.clearPlacement)
        self.regionList = qt.QListWidget()
        self.regionList.maximumHeight = 110
        self.regionList.connect("itemDoubleClicked(QListWidgetItem*)", self.jumpRegion)
        review.addRow("双击定位 / Navigate", self.regionList)
        self.button(review, "用放置点移动选中疑点 / Move selected marker", self.moveRegion)
        self.button(review, "删除选中疑点 / Remove selected marker", self.removeRegion)
        self.reviewBox = qt.QComboBox()
        review.addRow("复核项", self.reviewBox)
        self.reviewerEdit = qt.QLineEdit()
        self.reviewerEdit.placeholderText = "医生匿名ID"
        review.addRow("复核人", self.reviewerEdit)
        self.statusBox = qt.QComboBox()
        for text, value in (("已复核 / Reviewed", "reviewed"), ("不适用（说明原因） / N/A", "not_applicable"), ("仍不确定 / Uncertain", "uncertain")):
            self.statusBox.addItem(text, value)
        review.addRow("结论", self.statusBox)
        self.reviewNote = qt.QPlainTextEdit()
        self.reviewNote.maximumHeight = 70
        self.reviewNote.placeholderText = "写明检查部位与切片、是否保留真实缺损；不适用也需说明。"
        review.addRow("复核依据", self.reviewNote)
        self.button(review, "记录当前复核项 / Record review", self.recordReview)

        export = self.section("5. 质控与版本导出 / QC and versioning")
        self.button(export, "运行质控 / Run QC", self.runQC)
        self.outputEdit = qt.QLineEdit()
        export.addRow("导出根目录", self.outputEdit)
        self.button(export, "选择导出目录… / Browse", self.browseOutput)
        self.button(export, "另存新版本（含校验） / Export new version", self.exportVersion)
        self.report = qt.QPlainTextEdit()
        self.report.readOnly = True
        self.report.minimumHeight = 140
        export.addRow(self.report)
        self.statusLabel = qt.QLabel("未加载 / No review session")
        self.statusLabel.wordWrap = True
        self.layout.addWidget(self.statusLabel)
        self.layout.addStretch(1)
        self.refreshReviewItems()
        self.sceneObservers = [slicer.mrmlScene.AddObserver(slicer.mrmlScene.StartCloseEvent, self.onSceneClosing),
                               slicer.mrmlScene.AddObserver(slicer.mrmlScene.EndCloseEvent, self.onSceneClosed)]

    def section(self, title):
        box = ctk.ctkCollapsibleButton()
        box.text = title
        self.layout.addWidget(box)
        return qt.QFormLayout(box)

    def button(self, form, title, callback):
        button = qt.QPushButton(title)
        button.connect("clicked()", lambda: self.guard(callback))
        form.addRow(button)
        return button

    def guard(self, callback):
        try:
            callback()
        except Exception as exc:
            self.statusLabel.text = str(exc)
            slicer.util.errorDisplay(str(exc))

    def requireSession(self):
        if self.session is None:
            raise ValueError("请先载入已有精标 / Start a session first")
        return self.session

    def browse(self):
        value = qt.QFileDialog.getOpenFileName(slicer.util.mainWindow(), "Existing segmentation", "", "Slicer segmentation (*.seg.nrrd)")
        if value:
            self.segEdit.text = value

    def browseOutput(self):
        value = qt.QFileDialog.getExistingDirectory(slicer.util.mainWindow(), "Export versions to")
        if value:
            self.outputEdit.text = value

    def start(self):
        if self.draftProcess:
            raise ValueError("请等待初稿任务结束 / Wait for the draft task")
        if self.session:
            raise ValueError("请先导出并关闭当前会话 / Export and close the current session first")
        volume = self.volumeBox.currentNode()
        if not volume:
            raise ValueError("先在Slicer载入原始CBCT / Load the source image in Slicer first")
        self.session = ReviewSession(self.segEdit.text, volume, self.caseEdit.text.strip())
        self.pickNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "CBCTReview_PlaceOnePoint")
        self.placeWidget.setCurrentNode(self.pickNode)
        self.binding.checked = False
        self.attachSession()

    def browseJobs(self):
        path = qt.QFileDialog.getExistingDirectory(slicer.util.mainWindow(), "Local draft job directory")
        if path:
            self.jobRootEdit.text = path

    def startBackendProcess(self, job=None, preflight=False):
        if self.draftProcess:
            raise ValueError("已有任务运行中 / A backend task is already running")
        command, args = draft_bridge.backend_command(job, preflight, self.providerBox.currentData)
        self.draftMode = 'preflight' if preflight else 'inference'
        self.draftProcess = qt.QProcess()
        self.draftProcess.setProcessChannelMode(qt.QProcess.MergedChannels)
        self.draftProcess.connect('readyReadStandardOutput()', self.readDraftLog)
        self.draftProcess.connect('finished(int,QProcess::ExitStatus)', self.draftFinished)
        self.draftProcess.connect('errorOccurred(QProcess::ProcessError)', self.draftProcessError)
        self.draftProcess.start(command, args)

    def checkBackend(self):
        self.draftLog.clear()
        self.startBackendProcess(preflight=True)

    def generateDraft(self):
        if self.session or self.draftProcess:
            raise ValueError("先导出并关闭精修会话，或等待当前任务结束 / Close the current review/task first")
        volume = self.volumeBox.currentNode()
        if not volume or not self.draftConsent.checked:
            raise ValueError("选择CBCT并确认3D输入范围 / Select a CBCT and confirm independent 3D input scope")
        if not self.jobRootEdit.text.strip():
            raise ValueError("选择本地任务目录 / Choose a local job directory")
        self.statusLabel.text = "正在本地导出影像；随后运行固定预训练模型 / Preparing a local inference job"
        slicer.app.processEvents()
        self.draftJob = draft_bridge.prepare_job(volume, self.caseEdit.text.strip(), self.jobRootEdit.text, True, self.providerBox.currentData)
        self.pendingVolume = volume
        self.draftLog.setPlainText("任务 / Job: " + str(self.draftJob))
        self.startBackendProcess(self.draftJob)

    def readDraftLog(self):
        if self.draftProcess:
            raw = self.draftProcess.readAllStandardOutput()
            try:
                value = bytes(raw).decode('utf-8', errors='replace')
            except TypeError:
                value = str(raw)
            self.draftLog.appendPlainText(value)

    def draftProcessError(self, error):
        if self.draftProcess and error == qt.QProcess.FailedToStart:
            self.draftLog.appendPlainText("无法启动WSL后端；检查WSL发行版与隔离Python路径。")
            self.draftProcess = None

    def draftFinished(self, code, status):
        self.readDraftLog()
        mode = self.draftMode
        self.draftProcess = None
        if code != 0:
            self.statusLabel.text = "初稿未完成；未导入掩膜。请检查任务日志 / No completed draft imported"
            return
        if mode == 'preflight':
            self.statusLabel.text = "模型文件与GPU检查通过 / Deployment check passed"
            return
        if self.draftJob and (self.draftJob / 'CANCEL').exists():
            self.statusLabel.text = "已请求取消；结果未自动导入 / Cancel requested; no automatic import"
            return
        self.guard(lambda: self.attachDraftJob(self.draftJob, self.pendingVolume))

    def attachDraftJob(self, job, volume):
        if self.session or not volume or not volume.GetScene():
            raise ValueError("影像或会话已变化；请重新选择原图并导入已完成任务 / Image/session changed")
        self.session = draft_bridge.import_job(job, volume)
        self.caseEdit.text = self.session.case_id
        self.segEdit.text = str(self.session.original_path)
        self.pickNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsFiducialNode', 'CBCTReview_PlaceOnePoint')
        self.placeWidget.setCurrentNode(self.pickNode)
        self.binding.checked = False
        self.attachSession()
        self.statusLabel.text = "模型初稿已导入，尚未经医生确认；先核对影像与类别，再精修并复核三类区域。"

    def importCompletedJob(self):
        if self.session or self.draftProcess:
            raise ValueError("先关闭当前会话或任务 / Close the current session/task first")
        volume = self.volumeBox.currentNode()
        if not volume:
            raise ValueError("先加载对应CBCT / Load the matching CBCT")
        path = qt.QFileDialog.getExistingDirectory(slicer.util.mainWindow(), "Completed draft job")
        if path:
            self.attachDraftJob(Path(path), volume)

    def cancelDraft(self):
        if self.draftProcess:
            if self.draftMode == 'inference' and self.draftJob:
                (self.draftJob / 'CANCEL').write_text('User requested cancellation', encoding='utf-8')
                self.statusLabel.text = "已请求取消；后端将在阶段或网络前向边界停止，请等待任务退出。"
            else:
                self.draftProcess.terminate()

    def attachSession(self):
        session = self.requireSession()
        session.working.GetSegmentation().SetConversionParameter("Smoothing factor", "0.0")
        session.working.GetSegmentation().SetConversionParameter("Decimation factor", "0.0")
        self.editor.setSegmentationNode(session.working)
        self.editor.setSourceVolumeNode(session.volume)
        self.parameterNode.SetOverwriteMode(slicer.vtkMRMLSegmentEditorNode.OverwriteNone)
        self.editor.setReadOnly(not session.binding_confirmed)
        self.segmentBox.clear()
        for sid in session.before:
            storage = session.original_storage
            raw_value = "?"
            for key, value in storage.items():
                if key.endswith("_ID") and value == sid:
                    raw_value = storage.get(key[:-3] + "_LabelValue", "?")
            self.segmentBox.addItem(session.original_names[sid] + " [原像素值 " + raw_value + "]", sid)
        slicer.util.setSliceViewerLayers(background=session.volume)
        self.baselineToggle.checked = False
        self.surfaceToggle.checked = False
        self.refreshReviewItems()
        self.statusLabel.text = "已创建工作副本；需确认患者与影像对应后编辑。/ Confirm image binding to edit."

    def bind(self, checked):
        if self.session:
            try:
                if checked:
                    report = self.session.qc()
                    if any(x["outside_reference_voxels"] for x in report["segments"].values()):
                        raise ValueError("标注超出参考影像；先审查影像对应关系 / Masks outside reference image")
                self.session.binding_confirmed = bool(checked)
                self.editor.setReadOnly(not checked or self.baselineToggle.checked)
            except Exception as exc:
                self.binding.checked = False
                self.statusLabel.text = str(exc)

    def selectSegment(self, index):
        if not self.session or index < 0:
            return
        sid = self.segmentBox.itemData(index)
        self.editor.setCurrentSegmentID(sid)
        mapping = self.session.mappings[sid]
        self.roleBox.setCurrentIndex(self.roleBox.findData(mapping["role"]))
        self.fdiEdit.text = mapping["fdi"]
        self.mappingNote.text = mapping["note"]

    def confirmMapping(self):
        session = self.requireSession()
        session.set_mapping(self.segmentBox.currentData, self.roleBox.currentData, self.fdiEdit.text, self.mappingNote.text)
        self.statusLabel.text = "对应关系已确认；映射或掩膜变化后请重做关键区域复核。"

    def showBaseline(self, checked):
        if self.session:
            self.session.baseline.GetDisplayNode().SetVisibility(checked)
            self.session.working.GetDisplayNode().SetVisibility(not checked)
            self.editor.setReadOnly(checked or not self.session.binding_confirmed)
            if checked:
                self.editor.setActiveEffect(None)

    def showSurface(self, checked):
        if not self.session:
            return
        if checked:
            self.session.working.GetSegmentation().SetConversionParameter("Smoothing factor", "0.0")
            self.session.working.GetSegmentation().SetConversionParameter("Decimation factor", "0.0")
            self.session.working.CreateClosedSurfaceRepresentation()
        self.session.working.GetDisplayNode().SetVisibility3D(checked)

    def addRegion(self):
        session = self.requireSession()
        if self.pickNode.GetNumberOfControlPoints() != 1:
            raise ValueError("请在切片放置且仅放置一个点 / Place exactly one point")
        ras = [0.0, 0.0, 0.0]
        self.pickNode.GetNthControlPointPositionWorld(0, ras)
        session.add_region(self.categoryBox.currentData, self.segmentBox.currentData, ras, self.regionNote.text)
        self.pickNode.RemoveAllControlPoints()
        self.refreshReviewItems()

    def clearPlacement(self):
        self.requireSession()
        self.pickNode.RemoveAllControlPoints()

    def moveRegion(self):
        session = self.requireSession()
        item = self.regionList.currentItem()
        if not item or self.pickNode.GetNumberOfControlPoints() != 1:
            raise ValueError("选择疑点并放置一个新位置点 / Select a region and place one point")
        ras = [0.0, 0.0, 0.0]
        self.pickNode.GetNthControlPointPositionWorld(0, ras)
        session.move_region(item.data(qt.Qt.UserRole), ras, self.regionNote.text)
        self.pickNode.RemoveAllControlPoints()
        self.refreshReviewItems()

    def removeRegion(self):
        item = self.regionList.currentItem()
        if not item:
            raise ValueError("选择需要删除的疑点 / Select a marker")
        self.requireSession().remove_region(item.data(qt.Qt.UserRole))
        self.refreshReviewItems()

    def refreshReviewItems(self):
        self.reviewBox.clear()
        self.regionList.clear()
        for key in core.CATEGORIES:
            self.reviewBox.addItem(LABELS[key] + "（整体）", key)
        if self.session:
            for region in self.session.regions():
                title = LABELS[region["category"]] + " / " + self.session.original_names[region["segment_id"]] + " / " + region["note"]
                self.reviewBox.addItem(title, "region:" + region["id"])
                item = qt.QListWidgetItem(title)
                item.setData(qt.Qt.UserRole, region["id"])
                self.regionList.addItem(item)

    def jumpRegion(self, item):
        rid = item.data(qt.Qt.UserRole)
        for region in self.requireSession().regions():
            if region["id"] == rid:
                slicer.util.setSliceViewerLayers(background=self.session.volume)
                slicer.util.resetSliceViews()
                slicer.modules.markups.logic().JumpSlicesToLocation(*region["ras_mm"], True)
                self.reviewBox.setCurrentIndex(self.reviewBox.findData("region:" + rid))
                self.segmentBox.setCurrentIndex(self.segmentBox.findData(region["segment_id"]))

    def recordReview(self):
        self.requireSession().review(self.reviewBox.currentData, self.reviewerEdit.text,
                                     self.statusBox.currentData, self.reviewNote.toPlainText())
        self.statusLabel.text = "已记录。继续其他复核项；全部完成后运行质控。/ Review recorded."

    def runQC(self):
        report = self.requireSession().qc()
        self.report.setPlainText(json.dumps({"gate": report["gate"], "segments": report["segments"], "reviews": report["reviews"]}, ensure_ascii=False, indent=2))
        self.statusLabel.text = report["gate"]["status"] + " / 完成复核不等于训练准入或临床验证。"
        return report

    def exportVersion(self):
        if not self.outputEdit.text.strip():
            raise ValueError("选择导出根目录 / Select an output directory")
        directory, report = self.requireSession().export(self.outputEdit.text)
        self.report.setPlainText(json.dumps(report["gate"], ensure_ascii=False, indent=2))
        self.statusLabel.text = "已导出并回读校验 / Verified export: " + str(directory)

    def resume(self):
        session = self.requireSession()
        if session.has_unexported_content():
            raise ValueError("请先导出当前更改，再恢复其他版本 / Export current changes before restoring a version")
        directory = qt.QFileDialog.getExistingDirectory(slicer.util.mainWindow(), "Resume exported version")
        if directory:
            session.resume(directory)
            self.binding.checked = False
            self.attachSession()

    def closeSession(self):
        session = self.requireSession()
        report = session.qc()
        if not session.last_export:
            raise ValueError("请先另存新版本 / Export a version before closing")
        previous = json.loads((session.last_export / "review.json").read_text(encoding="utf-8"))
        if previous["revision"] != report["revision"] or previous["reviews"] != report["reviews"] or previous["gate"] != report["gate"]:
            raise ValueError("存在未导出的更改或复核记录 / Export recent changes before closing")
        self.editor.setSegmentationNode(None)
        self.editor.setSourceVolumeNode(None)
        self.placeWidget.setCurrentNode(None)
        if self.pickNode:
            slicer.mrmlScene.RemoveNode(self.pickNode)
        session.close()
        self.session = self.pickNode = None
        self.binding.checked = False
        self.segmentBox.clear()
        self.refreshReviewItems()
        self.statusLabel.text = "会话已关闭 / Session closed"

    def exit(self):
        self.editor.setActiveEffect(None)
        self.editor.removeViewObservations()

    def cleanup(self):
        self.cancelDraft()
        for tag in getattr(self, "sceneObservers", []):
            slicer.mrmlScene.RemoveObserver(tag)
        if hasattr(self, "editor"):
            self.editor.setActiveEffect(None)
            self.editor.setMRMLScene(None)

    def onSceneClosing(self, caller, event):
        self.cancelDraft()
        self.editor.setActiveEffect(None)
        self.editor.setSegmentationNode(None)
        self.editor.setSourceVolumeNode(None)
        self.placeWidget.setCurrentNode(None)
        self.session = self.pickNode = None
        self.binding.checked = False
        self.segmentBox.clear()
        self.refreshReviewItems()
        self.statusLabel.text = "场景已关闭；请从已导出版本恢复 / Scene closed; resume from an exported version"

    def onSceneClosed(self, caller, event):
        self.parameterNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentEditorNode", "CBCTReviewEditor")
        self.parameterNode.SetOverwriteMode(slicer.vtkMRMLSegmentEditorNode.OverwriteNone)
        self.editor.setMRMLSegmentEditorNode(self.parameterNode)
