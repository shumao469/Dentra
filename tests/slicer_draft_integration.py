"""Real preflight + synthetic draft exchange; no synthetic result is called model inference."""
from pathlib import Path
import json
import sys
import tempfile
import time
import traceback

import numpy as np
import SimpleITK as sitk
import slicer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from CBCTSegReviewLib import core, draft, draft_bridge
from CBCTSegReviewLib.slicer_adapter import ReviewSession, native_snapshots


def run():
    tests = []
    def check(condition, name):
        assert condition, name
        tests.append({'test': name, 'status': 'PASS'})
    with tempfile.TemporaryDirectory(prefix='CBCTDraft_') as td:
        volume = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLScalarVolumeNode', 'SYNTHETIC_E3D')
        volume.SetSpacing(0.25, 0.25, 0.5)
        volume.SetOrigin(10.0, -20.0, 5.0)
        data = np.zeros((20, 25, 30), dtype=np.int16)
        slicer.util.updateVolumeFromArray(volume, data)
        job = draft_bridge.prepare_job(volume, 'SYNTHETIC_01', td, True)
        request = json.loads((job / 'job.json').read_text())
        check(request['cohort'] == '3D_ANNOTATION_ONLY', 'scope admission is explicit')
        image = sitk.ReadImage(str(job / 'input.nrrd'))
        array = np.zeros_like(data, dtype=np.uint8)
        array[3:17, 4:20, 5] = 1
        array[8:12, 9:14, 5] = 0
        array[3:17, 4:20, 20] = 2
        array[6:8, 9:11, 6:9] = 3
        array[12:14, 9:11, 15:18] = 4
        array[4:16, 18, 19] = 5
        labels = sitk.GetImageFromArray(array)
        labels.CopyInformation(image)
        (job / 'result').mkdir()
        sitk.WriteImage(labels, str(job / 'result' / 'draft_labels.nrrd'), True)
        receipt = {'schema': 'cbct-pretrained-draft/0.2', 'status': 'MODEL_DRAFT', 'case_id': request['case_id'],
                   'cohort': '3D_ANNOTATION_ONLY', 'model_id': draft.MODEL_ID,
                   'model_manifest_sha256': core.digest_file(ROOT / 'config' / 'model_manifest.json'),
                   'protocol_sha256': request['protocol_sha256'], 'input_file_sha256': request['input_file_sha256'],
                   'source_volume_signature': request['source_volume']['signature'],
                   'draft_file_sha256': core.digest_file(job / 'result' / 'draft_labels.nrrd'),
                   'required_manual_review': list(core.CATEGORIES), 'mask_postprocessing': [],
                   'training_admission': False, 'clinical_validation': False,
                   'test_fixture': 'SYNTHETIC_EXCHANGE_ONLY_NOT_A_NETWORK_PREDICTION'}
        (job / 'result' / 'draft_receipt.json').write_text(json.dumps(receipt), encoding='utf-8')
        session = draft_bridge.import_job(job, volume)
        check(len(session.before) == 5, 'five original model classes retained separately')
        check(session.draft_provenance == receipt, 'machine provenance survives segmentation file roundtrip')
        check(session.qc()['source_kind'] == 'MODEL_DRAFT', 'model draft is never silently marked human truth')
        check(session.qc()['gate']['status'] == 'IN_PROGRESS', 'all human reviews initially pending')
        check(all(not m['fdi'] and not m['confirmed'] for m in session.mappings.values()), 'no FDI or role is auto-confirmed')
        snapshots = native_snapshots(session.working)
        check(int(snapshots['model_label_1']['array'].sum()) == int((array == 1).sum()), 'thin plate and synthetic hole survive import without cleanup')
        try:
            session.set_mapping('model_label_3', 'tooth', '11')
            raise AssertionError('aggregate tooth group was relabeled as FDI')
        except ValueError:
            check(True, 'aggregate teeth cannot be mislabeled as individual FDI instances')
        try:
            session.set_mapping('model_label_1', 'alveolar_bone')
            raise AssertionError('upper skull mislabeled as alveolar bone')
        except ValueError:
            check(True, 'whole jaw cannot be silently relabeled as alveolar bone')
        export, report = session.export(Path(td) / 'versions')
        check(report['draft_provenance'] == receipt and not report['gate']['training_admission'], 'refinement export retains model provenance and exclusion from training')
        session.close()
        reopened = ReviewSession(job / 'slicer_source' / 'model_draft.seg.nrrd', volume, request['case_id'])
        check(reopened.draft_provenance == receipt, 'manual source-file reopening retains machine origin')
        reopened.resume(export)
        check(reopened.qc()['source_kind'] == 'MODEL_DRAFT', 'version restore preserves machine origin')
        reopened.close()
        altered = data.copy()
        altered[0, 0, 0] = 1
        slicer.util.updateVolumeFromArray(volume, altered)
        try:
            draft_bridge.import_job(job, volume)
            raise AssertionError('draft accepted on changed image')
        except ValueError:
            check(True, 'changed image pixels prevent cross-image import')
        slicer.util.updateVolumeFromArray(volume, data)
        import CBCTSegReview
        import qt
        parent = slicer.qMRMLWidget()
        parent.setLayout(qt.QVBoxLayout())
        widget = CBCTSegReview.CBCTSegReviewWidget(parent)
        widget.setup()
        widget.volumeBox.setCurrentNode(volume)
        widget.checkBackend()
        deadline = time.monotonic() + 100
        while widget.draftProcess and time.monotonic() < deadline:
            slicer.app.processEvents()
            time.sleep(0.05)
        check(widget.draftProcess is None and 'passed' in widget.statusLabel.text.lower(), 'actual Qt-to-WSL preflight completes asynchronously')
        widget.draftMode = 'inference'
        widget.draftJob = job
        (job / 'CANCEL').write_text('Synthetic cancellation race test')
        widget.draftFinished(0, qt.QProcess.NormalExit)
        check(widget.session is None, 'late cancellation prevents automatic import even after process success')
        (job / 'CANCEL').unlink()
        widget.attachDraftJob(job, volume)
        check(widget.session.draft_provenance == receipt and not widget.binding.checked, 'Qt import enters pending refinement session')
        widget.editor.setReadOnly(True)
        widget.outputEdit.text = str(Path(td) / 'ui_versions')
        widget.exportVersion()
        widget.closeSession()
        widget.cleanup()
    return {'status': 'PASS', 'scope': 'synthetic exchange contract plus real local preflight; not model accuracy', 'tests': tests,
            'tested_source_sha256': {str(p.relative_to(ROOT)): core.digest_file(p) for p in ROOT.rglob('*.py') if '__pycache__' not in p.parts}}


try:
    payload = run()
    code = 0
except Exception:
    payload = {'status': 'FAIL', 'traceback': traceback.format_exc()}
    code = 1
(ROOT / 'validation').mkdir(exist_ok=True)
(ROOT / 'validation' / 'draft_integration.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
slicer.app.exit(code)
