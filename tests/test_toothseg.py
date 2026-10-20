import copy
import unittest
from pathlib import Path
import numpy as np
from CBCTSegReviewLib import draft, toothseg, core

ROOT = Path(__file__).resolve().parents[1]
MAPPING = {i + 1: str(q * 10 + t) for i, (q, t) in enumerate((q, t) for q in range(1, 5) for t in range(1, 9))}


class ToothSegTests(unittest.TestCase):
    def test_released_anatomical_names_map_to_verified_fdi(self):
        import json
        manifest = json.loads((ROOT / 'config/toothseg_manifest.json').read_text())
        labels = manifest['branches']['semantic']['dataset_labels']
        mapping = toothseg.class_to_fdi({'labels': labels})
        self.assertEqual(mapping, MAPPING)
        changed = dict(labels)
        changed['Unknown Incisor'] = changed.pop('Upper Right Central Incisor')
        with self.assertRaises(ValueError):
            toothseg.class_to_fdi({'labels': changed})

    def test_foreground_holes_and_tiny_fragments_preserved(self):
        raw = np.zeros((12, 12, 12), np.uint8)
        raw[2:10, 2:10, 2] = 2
        raw[3, 3, 2] = 1
        raw[8, 8, 2] = 1
        raw[5:7, 5:7, 2] = 0  # synthetic true opening: must stay empty
        raw[0, 0, 0] = 2     # one-voxel border-only fragment: must survive
        original = raw.copy()
        sem = np.zeros_like(raw)
        sem[raw > 0] = 1
        instances, rows = toothseg.assemble(raw, sem, MAPPING)
        np.testing.assert_array_equal(raw, original)
        np.testing.assert_array_equal(instances > 0, raw > 0)
        self.assertEqual(len(rows), 3)
        self.assertTrue(any(r['voxels'] == 1 and 'BORDER_ONLY_NO_CORE' in r['flags'] for r in rows))
        self.assertTrue(all('DUPLICATE_FDI_PROPOSAL' in r['flags'] for r in rows))
        self.assertTrue(all(not r['confirmed'] for r in rows))
        toothseg.validate_instances(instances, {'instances': rows})

    def test_background_and_ties_stay_unknown(self):
        raw = np.ones((2, 2, 2), np.uint8)
        sem = np.zeros_like(raw)
        sem.ravel()[:4] = 1
        _, rows = toothseg.assemble(raw, sem, MAPPING)
        self.assertEqual(rows[0]['proposed_fdi'], '')
        self.assertIn('BACKGROUND_OR_TIED_SEMANTICS', rows[0]['flags'])

    def test_disjoint_same_fdi_never_merge(self):
        raw = np.zeros((5, 5, 5), np.uint8)
        raw[0, 0, 0] = raw[4, 4, 4] = 1
        instances, rows = toothseg.assemble(raw, np.ones_like(raw), MAPPING)
        self.assertEqual(len(rows), 2)
        self.assertNotEqual(instances[0, 0, 0], instances[4, 4, 4])

    def test_unknown_labels_or_wrong_geometry_rejected(self):
        for raw, sem in [(np.ones((2, 2, 2), np.uint8) * 3, np.ones((2, 2, 2), np.uint8)),
                         (np.ones((2, 2, 2), np.uint8), np.ones((3, 2, 2), np.uint8))]:
            with self.assertRaises(ValueError):
                toothseg.assemble(raw, sem, MAPPING)

    def test_table_mismatch_and_autoconfirmation_rejected(self):
        instances, rows = toothseg.assemble(np.ones((2, 2, 2), np.uint8), np.ones((2, 2, 2), np.uint8), MAPPING)
        for key, value in [('voxels', 999), ('confirmed', True), ('proposed_fdi', '99')]:
            changed = copy.deepcopy(rows)
            changed[0][key] = value
            with self.assertRaises(ValueError):
                toothseg.validate_instances(instances, {'instances': changed})

    def test_frozen_protocol_and_independent_reviews(self):
        protocol, ph = draft.load_protocol(ROOT, toothseg.MODEL_ID)
        self.assertEqual(protocol['instance_assembly'], toothseg.ASSEMBLY)
        self.assertEqual(protocol['small_component_threshold'], 0)
        self.assertFalse(protocol['training'])
        self.assertEqual(set(protocol['required_manual_review']), set(core.CATEGORIES))
        request = {'case_id': 'G04_001', 'model_id': toothseg.MODEL_ID, 'protocol_sha256': ph,
                   'cohort': '3D_ANNOTATION_ONLY', 'source_confirmed': True, 'input_filename': 'input.nrrd'}
        with self.assertRaises(ValueError):
            draft.check_request(request, ph)
