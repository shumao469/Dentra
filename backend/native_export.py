"""Channel-streamed native-grid logits resampling and exact first-wins argmax.

Uses the UNCHANGED frozen resampling function independently per class. Avoids
materializing a C-channel native-grid float array. No mask postprocessing.
"""
import numpy as np
import torch


def native_argmax(logits, plans_manager, configuration_manager, label_manager, properties, check_cancel=None):
    from acvl_utils.cropping_and_padding.bounding_boxes import insert_crop_into_image
    if label_manager.has_regions:
        raise ValueError('Streamed export supports contiguous class logits, not regions')
    spacing_transposed = [properties['spacing'][i] for i in plans_manager.transpose_forward]
    current_spacing = configuration_manager.spacing if len(configuration_manager.spacing) == len(properties['shape_after_cropping_and_before_resampling']) else [spacing_transposed[0], *configuration_manager.spacing]
    target_spacing = [properties['spacing'][i] for i in plans_manager.transpose_forward]
    target = properties['shape_after_cropping_and_before_resampling']
    best = np.full(target, -np.inf, np.float32)
    labels = np.zeros(target, np.uint8)
    for channel in range(logits.shape[0]):
        if check_cancel:
            check_cancel()
        score = configuration_manager.resampling_fn_probabilities(logits[channel:channel + 1], target,
                                                                   current_spacing, target_spacing)
        if isinstance(score, torch.Tensor):
            score = score.cpu().numpy()
        score = score[0]
        if not np.isfinite(score).all():
            raise ValueError('Nonfinite resampled logits')
        update = score > best  # strict comparison preserves torch.argmax first-wins ties
        labels[update] = channel
        best[update] = score[update]
        del score, update
    del best
    restored = np.zeros(properties['shape_before_cropping'], dtype=np.uint8)
    restored = insert_crop_into_image(restored, labels, properties['bbox_used_for_cropping'])
    return restored.transpose(plans_manager.transpose_backward)
