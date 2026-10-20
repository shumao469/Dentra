"""Synthetic instance/FDI exchange and optional local inference import. No scoring."""
from pathlib import Path
import json
import os
import sys
import tempfile
import time
import traceback
import numpy as np
import SimpleITK as sitk
import slicer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from CBCTSegReviewLib import core, draft, toothseg, draft_bridge
from CBCTSegReviewLib.slicer_adapter import native_snapshots


def run():
    tests = []
    def check(value, name):
        assert value, name
        tests.append({'test': name, 'status': 'PASS'})
    with tempfile.TemporaryDirectory(prefix='DentraInstance_') as td:
        volume = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLScalarVolumeNode', 'SYNTHETIC_3D')
        volume.SetSpacing(0.25, 0.3, 0.5)
        data = np.zeros((16, 18, 20), np.int16)
        slicer.util.updateVolumeFromArray(volume, data)
        job = draft_bridge.prepare_job(volume, 'SYNTHETIC_INSTANCES', td, True, toothseg.MODEL_ID)
        request = json.loads((job / 'job.json').read_text())
        image = sitk.ReadImage(str(job / 'input.nrrd'))
        labels = np.zeros_like(data, np.uint32)
        labels[2:9, 2:8, 2] = 1
        labels[4:6, 4:6, 2] = 0
        labels[2:9, 2:8, 12] = 2
        labels[13, 14, 15] = 3
        result = job / 'result'
        result.mkdir()
        for filename, array in [('draft_labels.nrrd', labels), ('raw_border_core.nrrd', (labels > 0).astype(np.uint8)),
                                ('raw_semantic_labels.nrrd', (labels > 0).astype(np.uint8))]:
            output = sitk.GetImageFromArray(array)
            output.CopyInformation(image)
            sitk.WriteImage(output, str(result / filename), True)
        rows = [{'instance_id': i, 'proposed_fdi': '11' if i < 3 else '', 'voxels': int((labels == i).sum()),
                 'flags': ['DUPLICATE_FDI_PROPOSAL'] if i < 3 else ['BACKGROUND_OR_TIED_SEMANTICS'], 'confirmed': False}
                for i in (1, 2, 3)]
        receipt = {'status': 'MODEL_DRAFT', 'model_id': toothseg.MODEL_ID, 'case_id': request['case_id'],
                   'protocol_sha256': request['protocol_sha256'], 'model_manifest_sha256': core.digest_file(ROOT / 'config/toothseg_manifest.json'),
                   'source_volume_signature': request['source_volume']['signature'], 'input_file_sha256': request['input_file_sha256'],
                   'draft_file_sha256': core.digest_file(result / 'draft_labels.nrrd'),
                   'raw_semantic_sha256': core.digest_file(result / 'raw_semantic_labels.nrrd'),
                   'raw_border_core_sha256': core.digest_file(result / 'raw_border_core.nrrd'),
                   'instances': rows, 'instance_assembly': toothseg.ASSEMBLY, 'native_foreground_preserved': True,
                   'FDI': 'UNCONFIRMED_HARD_VOTE_PROPOSALS', 'required_manual_review': list(core.CATEGORIES),
                   'mask_postprocessing': [], 'training_admission': False, 'clinical_validation': False,
                   'test_fixture': 'SYNTHETIC_ONLY_NOT_MODEL_INFERENCE'}
        (result / 'draft_receipt.json').write_text(json.dumps(receipt), encoding='utf-8')
        session = draft_bridge.import_job(job, volume)
        check(len(session.before) == 3, 'same FDI and one-voxel unknown instances remain separate')
        check(all(not r['confirmed'] for r in session.mappings.values()), 'FDI proposals never auto-confirm')
        check(session.mappings['model_label_1']['fdi'] == '11', 'suggested FDI populates editable mapping')
        check(native_snapshots(session.working)['model_label_3']['array'].sum() == 1, 'one-voxel fragment survives Slicer roundtrip')
        session.set_mapping('model_label_1', 'tooth', '11', 'synthetic test')
        session.set_mapping('model_label_2', 'tooth', '11', 'synthetic duplicate')
        session.set_mapping('model_label_3', 'tooth', '13', 'synthetic unknown resolved')
        check('duplicate_fdi_mapping' in session.qc()['gate']['blockers'], 'duplicate confirmed FDI cannot complete review')
        session.set_mapping('model_label_2', 'tooth', '12', 'clinician correction fixture')
        check(session.draft_provenance['instances'][1]['proposed_fdi'] == '11', 'human correction never rewrites raw model proposal')
        try:
            session.set_mapping('model_label_1', 'alveolar_bone')
            raise AssertionError('tooth model was silently promoted to bone')
        except ValueError:
            check(True, 'tooth instance cannot masquerade as alveolar bone')
        saved, report = session.export(Path(td) / 'versions')
        check(not report['gate']['training_admission'] and not report['gate']['clinical_validation'], 'export keeps training and clinical gates closed')
        session.close()
        resumed = draft_bridge.import_job(job, volume)
        resumed.resume(saved)
        check(resumed.mappings['model_label_2']['fdi'] == '12', 'corrected FDI survives version restore')
        check(resumed.draft_provenance['instances'][1]['proposed_fdi'] == '11', 'source proposal survives restore separately')
        resumed.close()
        import CBCTSegReview
        import qt
        parent = slicer.qMRMLWidget()
        parent.setLayout(qt.QVBoxLayout())
        widget = CBCTSegReview.CBCTSegReviewWidget(parent)
        widget.setup()
        widget.providerBox.setCurrentIndex(1)
        widget.volumeBox.setCurrentNode(volume)
        widget.checkBackend()
        deadline = time.monotonic() + 100
        while widget.draftProcess and time.monotonic() < deadline:
            slicer.app.processEvents()
            time.sleep(0.05)
        check(widget.draftProcess is None and 'passed' in widget.statusLabel.text.lower(), 'ToothSeg selector dispatches actual asynchronous WSL preflight')
        widget.attachDraftJob(job, volume)
        check(widget.fdiEdit.text == '11' and not widget.session.mappings['model_label_1']['confirmed'], 'Qt displays editable FDI suggestion with pending confirmation')
        widget.outputEdit.text = str(Path(td) / 'ui_versions')
        widget.exportVersion()
        widget.closeSession()
        widget.cleanup()
        raw = sitk.ReadImage(str(result / 'raw_border_core.nrrd'))
        changed = sitk.GetArrayFromImage(raw)
        changed[0, 0, 0] = 1
        out = sitk.GetImageFromArray(changed)
        out.CopyInformation(raw)
        sitk.WriteImage(out, str(result / 'raw_border_core.nrrd'))
        try:
            draft_bridge.import_job(job, volume)
            raise AssertionError('changed raw branch accepted')
        except ValueError:
            check(True, 'raw-branch tampering prevents import')
    pilot = os.environ.get('DENTRA_PILOT_JOB')
    real_summary = None
    if pilot:
        job = Path(pilot)
        volume = slicer.util.loadVolume(str(job / 'input.nrrd'))
        session = draft_bridge.import_job(job, volume)
        receipt = session.draft_provenance
        check(len(session.before) == len(receipt['instances']), 'actual dual-branch draft imports every retained instance')
        check(all(not m['confirmed'] for m in session.mappings.values()), 'actual model FDI remains clinician-pending')
        output, report = session.export(job / 'review_versions')
        check(report['source_kind'] == 'MODEL_DRAFT', 'actual review export retains machine provenance')
        real_summary = {'instances': len(receipt['instances']), 'suggested_fdi': sum(bool(r['proposed_fdi']) for r in receipt['instances']),
                        'flagged_instances': sum(bool(r['flags']) for r in receipt['instances']),
                        'inference_seconds': receipt['inference_seconds'], 'accuracy_evaluation': False,
                        'native_foreground_preserved': receipt['native_foreground_preserved']}
        session.close()
    return {'status': 'PASS', 'tests': tests, 'real_inference_import': real_summary,
            'tested_source_sha256': {str(p.relative_to(ROOT)): core.digest_file(p) for p in ROOT.rglob('*.py') if '__pycache__' not in p.parts}}


try:
    payload, code = run(), 0
except Exception:
    payload, code = {'status': 'FAIL', 'traceback': traceback.format_exc()}, 1
(ROOT / 'validation').mkdir(exist_ok=True)
(ROOT / 'validation/toothseg_slicer.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
print(json.dumps(payload))
slicer.app.exit(code)
