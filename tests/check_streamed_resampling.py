"""Numerical equivalence of channel streaming vs frozen stock nnU-Net export.
Requires the inference environment, but not CUDA or a patient image.
"""
from pathlib import Path
import json
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'backend'))
from CBCTSegReviewLib import core
from native_export import native_argmax
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.inference.export_prediction import convert_predicted_logits_to_segmentation_with_correct_shape

config = json.loads((ROOT / 'config/local_backend.json').read_text())
manifest = json.loads((ROOT / 'config/toothseg_manifest.json').read_text())
tests = []
rng = np.random.default_rng(20261020)
for name, branch in manifest['branches'].items():
    folder = Path(config['toothseg_model_root']) / branch['folder']
    plans = PlansManager(json.loads((folder / 'plans.json').read_text()))
    manager = plans.get_configuration(branch['configuration'])
    dataset = json.loads((folder / 'dataset.json').read_text())
    labels = plans.get_label_manager(dataset)
    channels = labels.num_segmentation_heads
    for dtype in (torch.float16, torch.float32):
        for kind, target in [('upsample', [17, 19, 23]), ('downsample', [7, 9, 11]), ('same', [11, 13, 15]), ('tied', [17, 19, 23])]:
            data = torch.tensor(rng.normal(size=(channels, 11, 13, 15)), dtype=dtype)
            if kind == 'tied':
                data[:] = 0
            properties = {'spacing': [0.5, 0.25, 0.25], 'shape_after_cropping_and_before_resampling': target,
                          'shape_before_cropping': [v + 4 for v in target],
                          'bbox_used_for_cropping': [[2, v + 2] for v in target]}
            reference = convert_predicted_logits_to_segmentation_with_correct_shape(data.clone(), plans, manager, labels, properties, False, 4)
            streamed = native_argmax(data, plans, manager, labels, properties)
            assert np.array_equal(reference, streamed), (name, str(dtype), kind)
            tests.append({'branch': name, 'dtype': str(dtype), 'case': kind, 'exact_labels_equal': True,
                          'compared_voxels': int(reference.size)})
report = {'status': 'PASS', 'tests': tests, 'method': 'fixed_resampler_each_class_then_first_wins_argmax',
          'scope': 'synthetic numerical export equivalence; not model accuracy',
          'tested_source_sha256': {str(p.relative_to(ROOT)): core.digest_file(p) for p in ROOT.rglob('*.py') if '__pycache__' not in p.parts}}
(ROOT / 'validation/streamed_export_equivalence.json').write_text(json.dumps(report, indent=2))
print(json.dumps({'status': 'PASS', 'exact_label_comparisons': len(tests)}))
