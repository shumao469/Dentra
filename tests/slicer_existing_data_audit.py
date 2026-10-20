"""Read-only real-data smoke audit, run inside Slicer. Writes no source data.

Set CBCT_3D_ROOT to the local existing-annotation root; default E:/3D.
Images/segmentations are deliberately excluded from the deliverable.
"""
import collections
import json
import os
from pathlib import Path
import sys
import tempfile
import traceback

import numpy as np
import SimpleITK as sitk
import sitkUtils
import slicer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from CBCTSegReviewLib import core
from CBCTSegReviewLib.slicer_adapter import ReviewSession


def run():
    source = Path(os.environ.get('CBCT_3D_ROOT', 'E:/3D'))
    paths = sorted(source.rglob('*.seg.nrrd'))
    if not paths:
        raise ValueError('No existing segmentations found')
    source_hashes = {p: core.digest_file(p) for p in paths}
    volumes, results = {}, []
    with tempfile.TemporaryDirectory(prefix='CBCTSegReview_realdata_') as td:
        for number, path in enumerate(paths, 1):
            candidates = None
            for directory in path.parents:
                if directory == source.parent:
                    break
                found = sorted(directory.rglob('*.dcm'))
                parents = {p.parent for p in found}
                if len(parents) == 1:
                    candidates = found
                    break
            if candidates is None:
                raise ValueError('Could not unambiguously identify a local DICOM folder')
            dicom_directory = candidates[0].parent
            if dicom_directory not in volumes:
                series = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(dicom_directory))
                if len(series) != 1:
                    raise ValueError('Multiple DICOM series; manual binding required')
                files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(dicom_directory), series[0])
                if len(files) != len(candidates):
                    raise ValueError('Ambiguous DICOM series membership')
                reader = sitk.ImageSeriesReader()
                reader.SetFileNames(files)
                image = reader.Execute()
                node = sitkUtils.PushVolumeToSlicer(image, name='AUDIT_REFERENCE_' + str(len(volumes) + 1))
                volumes[dicom_directory] = node
            volume = volumes[dicom_directory]
            session = ReviewSession(path, volume, 'SAMPLE_' + str(number).zfill(2))
            try:
                report = session.qc()
                # No manual role/FDI confirmation or anatomical judgement is invented.
                assert report['gate']['status'] == 'IN_PROGRESS'
                directory, saved = session.export(td)
                results.append({'sample_id': session.case_id, 'source_basename': path.name,
                                'source_sha256': source_hashes[path], 'source_unchanged': core.digest_file(path) == source_hashes[path],
                                'reference_shape_kji': session.reference['shape_kji'],
                                'reference_signature': session.reference['signature'],
                                'segments': report['segments'], 'export_roundtrip': 'PASS',
                                'original_storage_metadata': report['original_storage_metadata'],
                                'anatomical_review': 'PENDING_DOCTOR', 'gate': report['gate']})
            finally:
                session.close()
    unchanged = all(core.digest_file(p) == digest for p, digest in source_hashes.items())
    assert unchanged
    return {'status': 'PASS', 'slicer_version': slicer.app.applicationVersion,
            'scope': 'read-only import/native-mask QC and roundtrip using associated local DICOM; not clinical validation',
            'source_count': len(paths), 'all_original_segmentations_unchanged': unchanged, 'samples': results,
            'tested_source_sha256': {str(p.relative_to(ROOT)): core.digest_file(p) for p in sorted(ROOT.rglob('*.py')) if '__pycache__' not in p.parts}}


try:
    payload = run()
    code = 0
except Exception:
    payload = {'status': 'FAIL', 'traceback': traceback.format_exc()}
    code = 1
(ROOT / 'validation').mkdir(exist_ok=True)
(ROOT / 'validation' / 'existing_data_audit.json').write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
print(json.dumps({'status': payload['status'], 'source_count': payload.get('source_count')}, ensure_ascii=False))
slicer.app.exit(code)
