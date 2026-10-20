"""Shared model-draft contract. No image editing or network communication."""
from pathlib import Path, PureWindowsPath
import json
import re

import numpy as np

from . import core

MODEL_ID = 'DentalSegmentator_v100_zenodo10829675'
LABELS = {1: 'upper_skull_maxilla', 2: 'mandible', 3: 'upper_teeth_group',
          4: 'lower_teeth_group', 5: 'mandibular_canal'}
MODEL_ROLES = {1: 'jaw_bone', 2: 'jaw_bone', 3: 'teeth_group', 4: 'teeth_group', 5: 'nerve_canal'}


def load_protocol(root, model_id=MODEL_ID):
    from .toothseg import MODEL_ID as TOOTHSEG
    if model_id not in (MODEL_ID, TOOTHSEG):
        raise ValueError('Unknown model provider')
    path = Path(root) / 'config' / ('toothseg_protocol.json' if model_id == TOOTHSEG else 'draft_protocol.json')
    value = json.loads(path.read_text(encoding='utf-8'))
    if value['model_id'] != model_id or value['output_mask_postprocessing'] or value['training']:
        raise ValueError('Unsupported model protocol')
    if value['smoothing'] or value['hole_filling'] or value['component_filtering']:
        raise ValueError('Mask-altering postprocessing is disabled')
    return value, core.digest_file(path)


def windows_to_wsl(path):
    value = PureWindowsPath(str(path))
    if not re.fullmatch('[A-Za-z]:', value.drive) or not value.is_absolute():
        raise ValueError('Use an absolute local Windows drive path; UNC/network paths are unsupported')
    return '/mnt/' + value.drive[0].lower() + '/' + '/'.join(value.parts[1:])


def check_request(request, protocol_hash):
    from .toothseg import MODEL_ID as TOOTHSEG
    core.safe_case_id(request['case_id'])
    if request.get('model_id') not in (MODEL_ID, TOOTHSEG) or request.get('protocol_sha256') != protocol_hash:
        raise ValueError('Model/protocol mismatch')
    if request.get('cohort') != '3D_ANNOTATION_ONLY' or request.get('source_confirmed') is not True:
        raise ValueError('Only explicitly confirmed independent 3D annotation inputs are admitted')
    if re.search(r'(^|[^a-z0-9])g04([^a-z0-9]|$)', request['case_id'], re.IGNORECASE):
        raise ValueError('G04 is isolated')
    if request.get('input_filename') != 'input.nrrd':
        raise ValueError('Unexpected input filename')


def check_geometry(shape_kji, ijk_to_ras, expected):
    if list(shape_kji) != expected['shape_kji'] or not np.allclose(ijk_to_ras, expected['ijk_to_ras'], atol=1e-5, rtol=0):
        raise ValueError('Image geometry mismatch')


def validate_labels(array):
    if not np.issubdtype(np.asarray(array).dtype, np.integer):
        raise ValueError('Model output must use integer class labels')
    values, counts = np.unique(array, return_counts=True)
    if any(int(v) not in (0, *LABELS) for v in values):
        raise ValueError('Unknown model class')
    if not any(int(v) != 0 and int(n) > 0 for v, n in zip(values, counts)):
        raise ValueError('Model returned an empty foreground')
    return {str(int(v)): int(n) for v, n in zip(values, counts)}


def validate_receipt(receipt, request, input_hash, mask_hash):
    from .toothseg import MODEL_ID as TOOTHSEG, ASSEMBLY
    if receipt.get('model_id') != request['model_id'] or receipt.get('protocol_sha256') != request['protocol_sha256']:
        raise ValueError('Draft model or protocol provenance mismatch')
    if receipt.get('input_file_sha256') != input_hash or receipt.get('draft_file_sha256') != mask_hash:
        raise ValueError('Draft source/output hash mismatch')
    if receipt.get('case_id') != request['case_id'] or receipt.get('source_volume_signature') != request['source_volume']['signature']:
        raise ValueError('Draft input identity mismatch')
    if receipt.get('status') != 'MODEL_DRAFT' or receipt.get('mask_postprocessing') != []:
        raise ValueError('Only unprocessed model drafts are accepted')
    if receipt.get('training_admission') is not False or receipt.get('clinical_validation') is not False:
        raise ValueError('Draft must remain unvalidated and excluded from training')
    if set(receipt.get('required_manual_review', [])) != set(core.CATEGORIES):
        raise ValueError('Critical-region manual review is missing')
    if request['model_id'] == TOOTHSEG:
        if receipt.get('instance_assembly') != ASSEMBLY or receipt.get('native_foreground_preserved') is not True:
            raise ValueError('Unverified or mask-altering instance assembly')
        if receipt.get('FDI') != 'UNCONFIRMED_HARD_VOTE_PROPOSALS':
            raise ValueError('FDI suggestions must remain pending')
