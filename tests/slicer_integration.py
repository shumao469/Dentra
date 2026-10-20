"""Run INSIDE Slicer. Only synthetic image/masks; never edits research source data."""
import json
from pathlib import Path
import sys
import tempfile
import traceback
import faulthandler
import shutil

import numpy as np
import slicer
import vtk

ROOT = Path(__file__).resolve().parents[1]
(ROOT / 'validation').mkdir(exist_ok=True)
_trace = open(ROOT / 'validation' / 'runtime_trace.log', 'w', encoding='utf-8')
faulthandler.dump_traceback_later(40, repeat=True, file=_trace)
sys.path.insert(0, str(ROOT))
from CBCTSegReviewLib import core
from CBCTSegReviewLib.slicer_adapter import ReviewSession, native_snapshots


def run():
    results = []
    def check(condition, name):
        assert condition, name
        results.append({'test': name, 'status': 'PASS'})
        (ROOT / 'validation' / 'progress.json').write_text(json.dumps(results), encoding='utf-8')
    slicer.mrmlScene.Clear(0)
    output = ROOT / 'validation'
    output.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='CBCTSegReview_') as td:
        td = Path(td)
        volume = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLScalarVolumeNode', 'PHANTOM')
        volume.SetSpacing(0.2, 0.2, 0.3)
        image = np.zeros((32, 40, 48), dtype=np.int16)
        slicer.util.updateVolumeFromArray(volume, image)
        segmentation = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLSegmentationNode')
        segmentation.CreateDefaultDisplayNodes()
        segmentation.SetReferenceImageGeometryParameterFromVolumeNode(volume)
        sid = segmentation.GetSegmentation().AddEmptySegment('bone', 'Synthetic bone', (0.8, 0.7, 0.4))
        mask = np.zeros_like(image, dtype=np.uint8)
        mask[5:27, 10:30, 15] = 1  # one voxel thin plate
        mask[13:18, 17:22, 15] = 0  # fenestration
        mask[5:11, 11:15, 15] = 0  # open margin defect
        slicer.util.updateSegmentBinaryLabelmapFromArray(mask, segmentation, sid, volume)
        path = td / 'original.seg.nrrd'
        assert slicer.util.saveNode(segmentation, str(path))
        source_hash = core.digest_file(path)
        session = ReviewSession(path, volume, 'PHANTOM')
        check(not session.baseline.GetStorageNode().GetFileName(), 'protected baseline has no writable original save target')
        check(not session.has_unexported_content(), 'fresh session has no unexported mask changes')
        check(int(session.before[sid]['array'].sum()) == int(mask.sum()), 'one-voxel plate and defects preserved on import')
        check(not session.qc()['gate']['training_admission'], 'never automatic training admission')
        check(session.qc()['gate']['status'] == 'IN_PROGRESS', 'unbound and unmapped import stays pending')
        session.set_mapping(sid, 'alveolar_bone', note='Synthetic phantom only')
        check(session.has_unexported_content(), 'unexported semantic mapping protected before restore')
        session.binding_confirmed = True
        ras = (3.0, 4.0, 4.5)
        rid = session.add_region('fenestration', sid, ras, 'Synthetic window at native plate')
        for key in list(core.CATEGORIES) + ['region:' + rid]:
            session.review(key, 'TEST_ONLY', 'reviewed', 'Synthetic slice review; not a clinical annotation')
        check(session.qc()['gate']['status'] == 'REVIEW_COMPLETE', 'category and region reviews complete')
        first, report = session.export(td / 'versions')
        check(not session.has_unexported_content(), 'exported session has no pending content')
        check(first.is_dir(), 'atomic new-version export')
        check(core.digest_file(path) == source_hash, 'source file unchanged after export')
        loaded = slicer.util.loadSegmentation(str(first / 'refined.seg.nrrd'))
        check(core.fingerprint(native_snapshots(loaded)[sid]) == core.fingerprint(session.before[sid]), 'exact native-mask round trip; defects not filled')
        slicer.mrmlScene.RemoveNode(loaded)
        after = mask.copy()
        after[15, 19, 15] = 1  # simulate accidentally filling one voxel of the window
        slicer.util.updateSegmentBinaryLabelmapFromArray(after, session.working, sid, volume)
        qc = session.qc()
        check(qc['segments'][sid]['delta']['added_voxels'] == 1, 'one added defect voxel detected')
        check(qc['gate']['status'] == 'IN_PROGRESS', 'edit invalidates every previous critical review')
        second, _ = session.export(td / 'versions')
        check(second != first and (first / 'review.json').is_file(), 'later export never overwrites earlier version')
        session.resume(first)
        check(not session.binding_confirmed, 'resume requires reconfirming image-patient binding')
        check(core.fingerprint(native_snapshots(session.working)[sid]) == core.fingerprint(session.before[sid]), 'resume restores prior unfilled window')
        previous_revision = session.state()[3]
        session.move_region(rid, (3.0, 4.1, 4.5), 'Corrected synthetic marker')
        check(session.state()[3] != previous_revision, 'moving review marker invalidates previous review revision')
        session.remove_region(rid)
        check(not session.regions(), 'controlled review marker removal')
        corrupt = td / 'corrupt_version'
        shutil.copytree(first, corrupt)
        (corrupt / 'review.json').write_text('{}', encoding='utf-8')
        try:
            session.resume(corrupt)
            raise AssertionError('corrupt version accepted')
        except ValueError as exc:
            check('checksum' in str(exc).lower(), 'resume rejects checksum corruption')
        transform = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLLinearTransformNode')
        session.working.SetAndObserveTransformNodeID(transform.GetID())
        try:
            session.qc()
            raise AssertionError('parent transform accepted')
        except ValueError as exc:
            check('transform' in str(exc).lower(), 'parent transform rejected; never auto hardened')
        session.working.SetAndObserveTransformNodeID(None)
        # Instantiate the actual Qt widget and exercise its load/bind/mapping controls.
        import CBCTSegReview
        parent = slicer.qMRMLWidget()
        parent.setLayout(__import__('qt').QVBoxLayout())
        widget = CBCTSegReview.CBCTSegReviewWidget(parent)
        widget.setup()
        widget.caseEdit.text = 'UI_PHANTOM'
        widget.segEdit.text = str(path)
        widget.volumeBox.setCurrentNode(volume)
        widget.start()
        widget.binding.checked = True
        check(widget.session.binding_confirmed, 'Qt UI source loading and binding')
        check(widget.parameterNode.GetOverwriteMode() == slicer.vtkMRMLSegmentEditorNode.OverwriteNone, 'editing cannot automatically overwrite another segment')
        check(not widget.editor.unorderedEffectsVisible, 'only explicit manual effects visible')
        widget.roleBox.setCurrentIndex(widget.roleBox.findData('alveolar_bone'))
        widget.mappingNote.text = 'Synthetic UI test'
        widget.confirmMapping()
        check(widget.session.mappings[sid]['confirmed'], 'Qt UI explicit semantic mapping')
        widget.editor.saveStateForUndo()
        slicer.util.updateSegmentBinaryLabelmapFromArray(after, widget.session.working, sid, volume)
        widget.editor.undo()
        check(core.fingerprint(native_snapshots(widget.session.working)[sid]) == core.fingerprint(widget.session.before[sid]), 'Segment Editor undo restores native thin plate and window')
        widget.surfaceToggle.checked = True
        check(core.fingerprint(native_snapshots(widget.session.working)[sid]) == core.fingerprint(widget.session.before[sid]), '3D surface display leaves binary labels unchanged')
        widget.pickNode.AddControlPoint(vtk.vtkVector3d(*ras))
        widget.regionNote.text = 'Synthetic UI review marker'
        widget.addRegion()
        check(len(widget.session.regions()) == 1, 'Qt UI point placement enters review list')
        widget.regionList.setCurrentRow(0)
        widget.pickNode.AddControlPoint(vtk.vtkVector3d(3.0, 4.2, 4.5))
        widget.moveRegion()
        check(abs(widget.session.regions()[0]['ras_mm'][1] - 4.2) < 1e-6, 'Qt UI marker fine adjustment')
        widget.regionList.setCurrentRow(0)
        widget.removeRegion()
        check(not widget.session.regions(), 'Qt UI marker removal')
        widget.outputEdit.text = str(td / 'ui_versions')
        widget.exportVersion()
        check(widget.session.last_export.is_dir(), 'Qt UI draft export')
        # Render the real widget, containing phantom data only, for UI inspection.
        parent.resize(760, 2000)
        parent.show()
        slicer.app.processEvents()
        parent.grab().save(str(output / 'ui_phantom.png'))
        widget.closeSession()
        slicer.mrmlScene.Clear(0)
        check(widget.session is None and widget.parameterNode.GetScene() is not None, 'scene-close observers reset editor safely')
        widget.cleanup()
        parent.hide()
        session.close()
        check(core.digest_file(path) == source_hash, 'source unchanged through entire workflow')
    return results


try:
    results = run()
    payload = {'status': 'PASS', 'slicer_version': slicer.app.applicationVersion, 'tests': results,
               'tested_source_sha256': {str(p.relative_to(ROOT)): core.digest_file(p) for p in sorted(ROOT.rglob('*.py')) if '__pycache__' not in p.parts}}
    code = 0
except Exception:
    payload = {'status': 'FAIL', 'traceback': traceback.format_exc()}
    code = 1
(ROOT / 'validation').mkdir(exist_ok=True)
(ROOT / 'validation' / 'slicer_integration.json').write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
print(json.dumps(payload, ensure_ascii=False))
faulthandler.cancel_dump_traceback_later()
_trace.close()
slicer.app.exit(code)
