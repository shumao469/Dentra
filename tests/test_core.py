import unittest
import tempfile
from pathlib import Path
import numpy as np

from CBCTSegReviewLib import core


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.mask = np.zeros((12, 13, 14), dtype=np.uint8)
        self.mask[2:10, 3:11, 4:12] = 1
        self.mask[5, 6, 4:12] = 0  # an open defect; the library must not alter it
        self.matrix = np.diag([0.2, 0.2, 0.3, 1.0])

    def test_snapshot_does_not_smooth_or_mutate(self):
        old = self.mask.copy()
        snap = core.snapshot(self.mask, self.matrix)
        np.testing.assert_array_equal(self.mask, old)
        self.assertEqual(int(snap['array'].sum()), int(old.sum()))
        self.assertEqual(int(snap['array'][3, 3].sum()), 0)

    def test_padding_invariance_with_translated_origin(self):
        a = core.snapshot(self.mask, self.matrix)
        padded = np.pad(self.mask, ((2, 0), (3, 0), (4, 0)))
        matrix = self.matrix.copy()
        matrix[:3, 3] = (matrix @ [-4, -3, -2, 1])[:3]
        b = core.snapshot(padded, matrix)
        self.assertEqual(core.fingerprint(a), core.fingerprint(b))
        self.assertEqual(core.delta(a, b)['added_voxels'], 0)

    def test_exact_add_remove(self):
        after = self.mask.copy()
        after[5, 6, 7] = 1
        after[2, 3, 4] = 0
        result = core.delta(core.snapshot(self.mask, self.matrix), core.snapshot(after, self.matrix))
        self.assertEqual(result['added_voxels'], 1)
        self.assertEqual(result['removed_voxels'], 1)

    def test_mask_motion_is_not_padding(self):
        a = core.snapshot(self.mask, self.matrix)
        matrix = self.matrix.copy()
        matrix[0, 3] += 0.2
        b = core.snapshot(self.mask, matrix)
        self.assertNotEqual(core.fingerprint(a), core.fingerprint(b))
        self.assertGreater(core.delta(a, b)['added_voxels'], 0)

    def test_grid_change_is_not_resampled(self):
        a = core.snapshot(self.mask, self.matrix)
        matrix = self.matrix.copy()
        matrix[0, 0] *= 2
        b = core.snapshot(self.mask, matrix)
        self.assertEqual(core.delta(a, b)['comparison'], 'GRID_CHANGED_NO_RESAMPLING')

    def test_empty_mask(self):
        a = core.snapshot(np.zeros_like(self.mask), self.matrix)
        b = core.snapshot(self.mask, self.matrix)
        self.assertEqual(core.delta(a, b)['added_voxels'], int(self.mask.sum()))

    def test_physical_containment(self):
        snap = core.snapshot(self.mask, self.matrix)
        qc = core.mask_qc(snap, self.matrix, self.mask.shape)
        self.assertEqual(qc['outside_reference_voxels'], 0)
        shifted = self.matrix.copy()
        shifted[0, 3] += 100
        qc = core.mask_qc(snap, shifted, self.mask.shape)
        self.assertEqual(qc['outside_reference_voxels'], int(self.mask.sum()))

    def test_different_sampling_can_be_physically_contained(self):
        snap = core.snapshot(self.mask, self.matrix)
        coarse = self.matrix.copy()
        coarse[:3, :3] *= 2
        qc = core.mask_qc(snap, coarse, self.mask.shape)
        self.assertFalse(qc['same_axis_sampling_as_volume'])
        self.assertEqual(qc['outside_reference_voxels'], 0)

    def test_invalid_affine(self):
        for matrix in (np.zeros((4, 4)), np.full((4, 4), np.nan)):
            with self.assertRaises(ValueError):
                core.snapshot(self.mask, matrix)

    def test_fdi_is_never_label_value(self):
        self.assertEqual(core.mapping_error({'role': 'tooth', 'fdi': '', 'confirmed': True}), 'tooth_requires_explicit_fdi')
        self.assertIsNone(core.mapping_error({'role': 'tooth', 'fdi': '11', 'confirmed': True}))
        self.assertEqual(core.mapping_error({'role': 'tooth', 'fdi': '99', 'confirmed': True}), 'tooth_requires_explicit_fdi')
        self.assertEqual(core.mapping_error({'role': 'alveolar_bone', 'fdi': '11', 'confirmed': True}), 'fdi_only_for_tooth')

    def test_nrrd_metadata_separator(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / 'mask.seg.nrrd'
            p.write_bytes(b'NRRD0004\nspace: left-posterior-superior\nSegment0_Name:=Segment_11\nSegment0_LabelValue:=24\n\n\x1f\x8b')
            fields = core.nrrd_metadata(p)
            self.assertEqual(fields['Segment0_LabelValue'], '24')
            self.assertEqual(fields['Segment0_Name'], 'Segment_11')

    def ready(self):
        qc = {'s': core.mask_qc(core.snapshot(self.mask, self.matrix), self.matrix, self.mask.shape)}
        mappings = {'s': {'role': 'alveolar_bone', 'fdi': '', 'confirmed': True}}
        revision = 'current'
        reviews = {key: core.review_record(revision, 'R1', 'reviewed', 'Checked native orthogonal slices') for key in core.CATEGORIES}
        return qc, mappings, reviews, revision

    def test_ready_is_not_training_admission(self):
        qc, mappings, reviews, rev = self.ready()
        result = core.release_check(qc, mappings, reviews, [], rev, True)
        self.assertEqual(result['status'], 'REVIEW_COMPLETE')
        self.assertFalse(result['training_admission'])
        self.assertFalse(result['clinical_validation'])

    def test_edit_invalidates_reviews(self):
        qc, mappings, reviews, rev = self.ready()
        self.assertEqual(core.release_check(qc, mappings, reviews, [], 'changed', True)['status'], 'IN_PROGRESS')

    def test_segment_rename_invalidates_review_revision(self):
        a = core.revision_id({'s': 'h'}, {}, [], 'v', {'s': 'name'})
        b = core.revision_id({'s': 'h'}, {}, [], 'v', {'s': 'new name'})
        self.assertNotEqual(a, b)

    def test_added_bone_is_a_dedicated_warning(self):
        qc, mappings, reviews, rev = self.ready()
        qc['s']['delta'] = {'added_voxels': 1, 'removed_voxels': 0, 'comparison': 'native_lattice'}
        result = core.release_check(qc, mappings, reviews, [], rev, True)
        self.assertIn('s:bone_added_check_thin_plate_and_true_defect_margins', result['warnings'])

    def test_marker_requires_own_review(self):
        qc, mappings, reviews, rev = self.ready()
        result = core.release_check(qc, mappings, reviews, [{'id': 'r1'}], rev, True)
        self.assertIn('region:r1:review_pending_or_stale', result['blockers'])

    def test_uncertain_and_unbound_cannot_complete(self):
        qc, mappings, reviews, rev = self.ready()
        reviews['thin_plate']['status'] = 'uncertain'
        result = core.release_check(qc, mappings, reviews, [], rev, False)
        self.assertIn('reference_patient_binding_unconfirmed', result['blockers'])
        self.assertEqual(result['status'], 'IN_PROGRESS')

    def test_duplicate_fdi(self):
        qc, mappings, reviews, rev = self.ready()
        qc['s2'] = qc['s']
        mappings = {sid: {'role': 'tooth', 'fdi': '11', 'confirmed': True} for sid in qc}
        self.assertIn('duplicate_fdi_mapping', core.release_check(qc, mappings, reviews, [], rev, True)['blockers'])

    def test_reviewer_and_note_required(self):
        with self.assertRaises(ValueError):
            core.review_record('a', '', 'reviewed', 'note')
        with self.assertRaises(ValueError):
            core.review_record('a', 'R1', 'not_applicable', '')

    def test_no_auto_generation(self):
        with self.assertRaises(NotImplementedError):
            core.DraftProvider().generate(None)

    def test_pseudonym_path_safety(self):
        self.assertEqual(core.safe_case_id('CASE_01'), 'CASE_01')
        for value in ('../x', 'a/b', 'Name with spaces', ''):
            with self.assertRaises(ValueError):
                core.safe_case_id(value)


if __name__ == '__main__':
    unittest.main()
