"""Slicer-side local-file bridge. No torch/nnU-Net installation inside Slicer."""
from pathlib import Path
import datetime
import json
import re
import uuid

import numpy as np
import SimpleITK as sitk
import sitkUtils
import slicer

from . import core, draft, toothseg
from .slicer_adapter import ReviewSession, native_snapshots, volume_info

ROOT = Path(__file__).resolve().parents[1]


def prepare_job(volume, case_id, output_root, source_confirmed, model_id=draft.MODEL_ID):
    protocol, protocol_hash = draft.load_protocol(ROOT, model_id)
    reference = volume_info(volume)
    if np.prod(reference['shape_kji']) > protocol['maximum_input_voxels']:
        raise ValueError('Volume exceeds the frozen single-job resource limit')
    storage = volume.GetStorageNode()
    paths = [volume.GetName() or '', storage.GetFileName() or '' if storage else '']
    if any(re.search(r'(^|[^a-z0-9])g04([^a-z0-9]|$)', name, re.IGNORECASE) for name in paths):
        raise ValueError('G04 is isolated')
    request = {'model_id': model_id, 'case_id': core.safe_case_id(case_id),
               'cohort': '3D_ANNOTATION_ONLY', 'source_confirmed': bool(source_confirmed),
               'protocol_sha256': protocol_hash, 'source_volume': reference, 'input_filename': 'input.nrrd'}
    draft.check_request(request, protocol_hash)
    root = Path(output_root).resolve()
    draft.windows_to_wsl(root)  # fail before exporting if WSL cannot access the local path
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    job = root / (case_id + '_' + stamp + '_' + uuid.uuid4().hex[:10])
    job.mkdir(parents=True, exist_ok=False)
    # Pull/export does not change the original MRML volume storage target.
    image = sitkUtils.PullVolumeFromSlicer(volume)
    sitk.WriteImage(image, str(job / 'input.nrrd'), True)
    request['input_file_sha256'] = core.digest_file(job / 'input.nrrd')
    (job / 'job.json').write_text(json.dumps(request, indent=2), encoding='utf-8')
    return job


def backend_command(job=None, preflight=False, model_id=draft.MODEL_ID):
    config = json.loads((ROOT / 'config' / 'local_backend.json').read_text(encoding='utf-8'))
    if job:
        model_id = json.loads((Path(job) / 'job.json').read_text(encoding='utf-8'))['model_id']
    script = 'toothseg_infer.py' if model_id == toothseg.MODEL_ID else 'infer.py'
    args = ['-d', config['distribution'], '--', config['python'], draft.windows_to_wsl(ROOT / 'backend' / script)]
    if preflight:
        args += ['--preflight']
    else:
        args += ['--job', draft.windows_to_wsl(job)]
    return 'wsl.exe', args


def import_job(job, volume):
    job = Path(job).resolve(strict=True)
    request = json.loads((job / 'job.json').read_text(encoding='utf-8'))
    is_toothseg = request['model_id'] == toothseg.MODEL_ID
    _, protocol_hash = draft.load_protocol(ROOT, request['model_id'])
    draft.check_request(request, protocol_hash)
    if volume_info(volume)['signature'] != request['source_volume']['signature']:
        raise ValueError('Selected image changed; draft cannot be attached to this image')
    result = job / 'result'
    receipt = json.loads((result / 'draft_receipt.json').read_text(encoding='utf-8'))
    draft.validate_receipt(receipt, request, core.digest_file(job / 'input.nrrd'), core.digest_file(result / 'draft_labels.nrrd'))
    manifest_hash = core.digest_file(ROOT / 'config' / ('toothseg_manifest.json' if is_toothseg else 'model_manifest.json'))
    if receipt.get('model_manifest_sha256') != manifest_hash:
        raise ValueError('Pretrained weights manifest differs from the recorded model draft')
    image = sitk.ReadImage(str(result / 'draft_labels.nrrd'))
    array = sitk.GetArrayFromImage(image)
    if is_toothseg:
        toothseg.validate_instances(array, receipt)
        for filename, key in (('raw_semantic_labels.nrrd', 'raw_semantic_sha256'), ('raw_border_core.nrrd', 'raw_border_core_sha256')):
            if core.digest_file(result / filename) != receipt.get(key):
                raise ValueError('ToothSeg raw branch output hash mismatch')
        raw = sitk.ReadImage(str(result / 'raw_border_core.nrrd'))
        if (raw.GetSize() != image.GetSize() or raw.GetDirection() != image.GetDirection()
                or raw.GetSpacing() != image.GetSpacing() or raw.GetOrigin() != image.GetOrigin()
                or not np.array_equal(sitk.GetArrayFromImage(raw) > 0, array > 0)):
            raise ValueError('Native ToothSeg foreground conservation/geometry failed on import')
    else:
        draft.validate_labels(array)
    matrix = np.eye(4)
    matrix[:3, :3] = np.diag([-1, -1, 1]) @ np.asarray(image.GetDirection()).reshape(3, 3) @ np.diag(image.GetSpacing())
    matrix[:3, 3] = np.diag([-1, -1, 1]) @ image.GetOrigin()
    draft.check_geometry(array.shape, matrix, request['source_volume'])
    source = job / 'slicer_source'
    if source.exists():
        saved = json.loads((source / 'source.json').read_text(encoding='utf-8'))
        if saved['draft_receipt_sha256'] != core.digest_file(result / 'draft_receipt.json') or saved['segmentation_sha256'] != core.digest_file(source / 'model_draft.seg.nrrd'):
            raise ValueError('Stored Slicer draft source changed')
    else:
        partial = job / '.partial_slicer_source'
        partial.mkdir(exist_ok=False)
        node = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLSegmentationNode', request['case_id'] + '_MODEL_DRAFT')
        try:
            node.CreateDefaultDisplayNodes()
            node.SetReferenceImageGeometryParameterFromVolumeNode(volume)
            colors = {1: (0.9, 0.8, 0.6), 2: (0.6, 0.8, 0.9), 3: (0.9, 0.9, 0.9), 4: (0.7, 0.9, 0.7), 5: (1.0, 0.5, 0.3)}
            labels = {r['instance_id']: 'Instance_%04d_FDI_%s_PENDING' % (r['instance_id'], r['proposed_fdi'] or 'unknown') for r in receipt['instances']} if is_toothseg else draft.LABELS
            for label, name in labels.items():
                binary = array == label
                if not binary.any():
                    continue  # absent classes may be outside the source field of view
                color = colors.get(label, ((label * 0.618) % 0.8 + 0.2, (label * 0.382) % 0.8 + 0.2, (label * 0.271) % 0.8 + 0.2))
                sid = node.GetSegmentation().AddEmptySegment('model_label_' + str(label), name, color)
                slicer.util.updateSegmentBinaryLabelmapFromArray(binary.astype(np.uint8), node, sid, volume)
                segment = node.GetSegmentation().GetSegment(sid)
                segment.SetTag('CBCT.SourceKind', 'MODEL_DRAFT')
                segment.SetTag('CBCT.ModelLabel', str(label))
                segment.SetTag('CBCT.DraftReceiptJSON', json.dumps(receipt, separators=(',', ':')))
            before = native_snapshots(node)
            if not slicer.util.saveNode(node, str(partial / 'model_draft.seg.nrrd')):
                raise RuntimeError('Slicer draft source write failed')
            readback = slicer.util.loadSegmentation(str(partial / 'model_draft.seg.nrrd'))
            try:
                after = native_snapshots(readback)
                if set(before) != set(after) or any(core.fingerprint(before[s]) != core.fingerprint(after[s]) for s in before):
                    raise RuntimeError('Slicer draft roundtrip changed voxels')
            finally:
                slicer.mrmlScene.RemoveNode(readback)
            saved = {'draft_receipt_sha256': core.digest_file(result / 'draft_receipt.json'),
                     'segmentation_sha256': core.digest_file(partial / 'model_draft.seg.nrrd')}
            (partial / 'source.json').write_text(json.dumps(saved, indent=2), encoding='utf-8')
            partial.rename(source)
        finally:
            slicer.mrmlScene.RemoveNode(node)
    session = ReviewSession(source / 'model_draft.seg.nrrd', volume, request['case_id'])
    if session.draft_provenance != receipt:
        session.close()
        raise RuntimeError('Model provenance did not survive Slicer segmentation save/load')
    for sid in session.before:
        label = int(sid.rsplit('_', 1)[1])
        if is_toothseg:
            row = next(r for r in receipt['instances'] if r['instance_id'] == label)
            session.mappings[sid] = {'role': 'tooth', 'fdi': row['proposed_fdi'], 'confirmed': False,
                                     'note': 'MODEL PROPOSAL ONLY; ' + ', '.join(row['flags'])}
        else:
            session.mappings[sid] = {'role': draft.MODEL_ROLES[label], 'fdi': '', 'confirmed': False,
                                     'note': 'Model class proposal only; confirm aggregate semantics manually'}
    return session
