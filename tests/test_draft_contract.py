import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from CBCTSegReviewLib import core, draft

ROOT = Path(__file__).resolve().parents[1]


class DraftContractTests(unittest.TestCase):
    def setUp(self):
        self.protocol, self.protocol_hash = draft.load_protocol(ROOT)
        self.request = {'case_id': 'DRAFT_001', 'model_id': draft.MODEL_ID,
                        'cohort': '3D_ANNOTATION_ONLY', 'source_confirmed': True,
                        'input_filename': 'input.nrrd', 'protocol_sha256': self.protocol_hash,
                        'source_volume': {'signature': 'volume-hash'}}
        self.receipt = {'model_id': draft.MODEL_ID, 'protocol_sha256': self.protocol_hash,
                        'input_file_sha256': 'input-hash', 'draft_file_sha256': 'draft-hash',
                        'case_id': 'DRAFT_001', 'source_volume_signature': 'volume-hash',
                        'status': 'MODEL_DRAFT', 'mask_postprocessing': [],
                        'required_manual_review': list(core.CATEGORIES),
                        'training_admission': False, 'clinical_validation': False}

    def test_frozen_protocol_no_mask_cleanup(self):
        self.assertEqual(self.protocol['output_mask_postprocessing'], [])
        self.assertFalse(self.protocol['FDI_numbering'])
        self.assertFalse(self.protocol['D_F_diagnosis'])
        draft.check_request(self.request, self.protocol_hash)

    def test_no_scope_confirmation_is_rejected(self):
        self.request['source_confirmed'] = False
        with self.assertRaises(ValueError):
            draft.check_request(self.request, self.protocol_hash)

    def test_G04_case_rejected(self):
        self.request['case_id'] = 'G04_001'
        with self.assertRaises(ValueError):
            draft.check_request(self.request, self.protocol_hash)

    def test_protocol_substitution_rejected(self):
        with self.assertRaises(ValueError):
            draft.check_request(self.request, 'different')

    def test_input_path_escape_rejected(self):
        self.request['input_filename'] = '../input.nrrd'
        with self.assertRaises(ValueError):
            draft.check_request(self.request, self.protocol_hash)

    def test_windows_paths_are_argument_data(self):
        self.assertEqual(draft.windows_to_wsl(r'D:\test folder\a$(x).nrrd'), '/mnt/d/test folder/a$(x).nrrd')
        with self.assertRaises(ValueError):
            draft.windows_to_wsl(r'\\server\share\image.nrrd')

    def test_invalid_output_classes_rejected(self):
        for array in (np.zeros((2, 2, 2), np.uint8), np.full((2, 2, 2), 11, np.uint8), np.ones((2, 2, 2), np.float32)):
            with self.assertRaises(ValueError):
                draft.validate_labels(array)

    def test_label_validation_never_fills_defects(self):
        array = np.zeros((10, 10, 10), np.uint8)
        array[2:8, 2:8, 5] = 1
        array[4:6, 4:6, 5] = 0
        original = array.copy()
        counts = draft.validate_labels(array)
        np.testing.assert_array_equal(array, original)
        self.assertEqual(counts['1'], 32)

    def test_receipt_requires_pending_machine_origin(self):
        draft.validate_receipt(self.receipt, self.request, 'input-hash', 'draft-hash')
        for key, value in (('status', 'GROUND_TRUTH'), ('training_admission', True),
                           ('mask_postprocessing', ['fill_holes']), ('required_manual_review', [])):
            changed = copy.deepcopy(self.receipt)
            changed[key] = value
            with self.assertRaises(ValueError):
                draft.validate_receipt(changed, self.request, 'input-hash', 'draft-hash')

    def test_cross_image_receipt_rejected(self):
        with self.assertRaises(ValueError):
            draft.validate_receipt(self.receipt, self.request, 'different-image-hash', 'draft-hash')

    def test_geometry_mismatch_rejected(self):
        expected = {'shape_kji': [4, 5, 6], 'ijk_to_ras': np.eye(4).tolist()}
        draft.check_geometry((4, 5, 6), np.eye(4), expected)
        with self.assertRaises(ValueError):
            draft.check_geometry((4, 6, 5), np.eye(4), expected)
        shifted = np.eye(4)
        shifted[0, 3] = 1
        with self.assertRaises(ValueError):
            draft.check_geometry((4, 5, 6), shifted, expected)

    def test_aggregate_teeth_roles_do_not_require_FDI(self):
        for role in ('jaw_bone', 'teeth_group', 'nerve_canal'):
            self.assertIsNone(core.mapping_error({'role': role, 'fdi': '', 'confirmed': True}))
            self.assertEqual(core.mapping_error({'role': role, 'fdi': '11', 'confirmed': True}), 'fdi_only_for_tooth')

    def test_jaw_additions_require_defect_warning(self):
        mask = np.ones((2, 2, 2), np.uint8)
        qc = {'jaw': core.mask_qc(core.snapshot(mask, np.eye(4)), np.eye(4), (4, 4, 4))}
        qc['jaw']['delta'] = {'added_voxels': 1, 'comparison': 'native_lattice'}
        result = core.release_check(qc, {'jaw': {'role': 'jaw_bone', 'fdi': '', 'confirmed': True}}, {}, [], 'a', True)
        self.assertIn('jaw:bone_added_check_thin_plate_and_true_defect_margins', result['warnings'])


if __name__ == '__main__':
    unittest.main()
