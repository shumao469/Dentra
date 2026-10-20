"""Offline single-volume pretrained nnU-Net inference. Never trains or postprocesses masks."""
from pathlib import Path
import argparse
import datetime
import hashlib
import importlib.metadata
import json
import os
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from CBCTSegReviewLib import core, draft


class Cancelled(RuntimeError):
    pass


def cancelled(job):
    if (job / 'CANCEL').exists():
        raise Cancelled('Cancellation requested; no completed draft is published')


def configuration():
    return json.loads((ROOT / 'config' / 'local_backend.json').read_text(encoding='utf-8'))


def model_files():
    config = configuration()
    manifest = json.loads((ROOT / 'config' / 'model_manifest.json').read_text(encoding='utf-8'))
    model_root = Path(config['model_root']).resolve(strict=True)
    for name, expected in manifest['files_sha256'].items():
        path = (model_root / name).resolve(strict=True)
        if not path.is_relative_to(model_root) or core.digest_file(path) != expected:
            raise ValueError('Pretrained model file hash mismatch: ' + name)
    model = model_root / manifest['training_output_relative']
    return model, manifest


def preflight():
    import torch
    protocol, protocol_hash = draft.load_protocol(ROOT)
    model, manifest = model_files()
    if importlib.metadata.version('nnunetv2') != '2.8.1':
        raise ValueError('Use the pinned nnunetv2==2.8.1 environment')
    gpu = int(protocol['gpu'].split(':')[1])
    if not torch.cuda.is_available() or gpu >= torch.cuda.device_count():
        raise ValueError('Configured CUDA GPU unavailable')
    torch.cuda.set_device(gpu)
    free, total = torch.cuda.mem_get_info(gpu)
    if free < protocol['minimum_free_gpu_gib'] * 1024**3:
        raise ValueError('Insufficient free GPU memory; do not change model parameters to force the run')
    return {'status': 'READY', 'python': sys.executable, 'torch': torch.__version__,
            'nnunetv2': importlib.metadata.version('nnunetv2'), 'gpu': gpu,
            'gpu_name': torch.cuda.get_device_name(gpu), 'gpu_free_bytes': free, 'gpu_total_bytes': total,
            'model_id': draft.MODEL_ID, 'model_manifest_sha256': core.digest_file(ROOT / 'config' / 'model_manifest.json'),
            'protocol_sha256': protocol_hash, 'weights_verified': True,
            'training': False, 'postprocessing': []}


def run(job):
    import numpy as np
    import SimpleITK as sitk
    import torch

    job = job.resolve(strict=True)
    if (job / 'result').exists() or (job / 'FAILED.json').exists():
        raise ValueError('Create a new job directory; never overwrite a prior run')
    with (job / 'RUNNING').open('x') as stream:
        stream.write(str(os.getpid()))
    started = time.monotonic()
    request = json.loads((job / 'job.json').read_text(encoding='utf-8'))
    protocol, protocol_hash = draft.load_protocol(ROOT)
    draft.check_request(request, protocol_hash)
    info = preflight()
    cancelled(job)
    input_path = job / 'input.nrrd'
    input_hash = core.digest_file(input_path)
    if input_hash != request['input_file_sha256']:
        raise ValueError('Input file changed')
    image = sitk.ReadImage(str(input_path))
    array = sitk.GetArrayFromImage(image)
    if array.ndim != 3 or array.size > protocol['maximum_input_voxels'] or not np.isfinite(array).all():
        raise ValueError('Input volume is invalid or exceeds frozen resource limit')
    matrix = np.eye(4)
    matrix[:3, :3] = np.diag([-1, -1, 1]) @ np.asarray(image.GetDirection()).reshape(3, 3) @ np.diag(image.GetSpacing())
    matrix[:3, 3] = np.diag([-1, -1, 1]) @ image.GetOrigin()
    draft.check_geometry(array.shape, matrix, request['source_volume'])
    if hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast('B')).hexdigest() != request['source_volume']['pixel_sha256']:
        raise ValueError('Exported input pixels differ from the selected Slicer volume')
    os.environ['nnUNet_compile'] = 'false'
    os.environ['nnUNet_raw'] = str(job / 'unused_raw')
    os.environ['nnUNet_preprocessed'] = str(job / 'unused_preprocessed')
    os.environ['nnUNet_results'] = str(job / 'unused_results')
    torch.set_num_threads(protocol['cpu_threads'])
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    predictor = nnUNetPredictor(tile_step_size=protocol['tile_step_size'], use_gaussian=True,
                                use_mirroring=protocol['use_checkpoint_mirroring_axes'],
                                perform_everything_on_device=True, device=torch.device(protocol['gpu']),
                                verbose=False, verbose_preprocessing=False, allow_tqdm=False)
    model, manifest = model_files()
    predictor.initialize_from_trained_model_folder(str(model), use_folds=manifest['folds'], checkpoint_name=manifest['checkpoint_name'])
    cancelled(job)
    tiles = [0]
    def tile_check(*args):
        cancelled(job)
        tiles[0] += 1
        if tiles[0] == 1 or tiles[0] % 20 == 0:
            print(json.dumps({'stage': 'inference', 'forward_calls': tiles[0]}), flush=True)
    handle = predictor.network.register_forward_pre_hook(tile_check)
    properties = {'spacing': list(image.GetSpacing())[::-1]}
    print(json.dumps({'stage': 'preprocessing', 'input_shape_kji': list(array.shape)}), flush=True)
    try:
        with torch.inference_mode():
            result = predictor.predict_single_npy_array(array[np.newaxis].astype(np.float32), properties,
                                                        None, None, False)
    finally:
        handle.remove()
    cancelled(job)
    segmentation = result[0] if isinstance(result, tuple) else result
    counts = draft.validate_labels(segmentation)
    draft.check_geometry(segmentation.shape, matrix, request['source_volume'])
    stage = job / '.partial_result'
    stage.mkdir(exist_ok=False)
    mask = sitk.GetImageFromArray(segmentation.astype(np.uint8))
    mask.CopyInformation(image)
    sitk.WriteImage(mask, str(stage / 'draft_labels.nrrd'), True)
    # Re-open file geometry and voxel values; no interpolation or mask cleanup.
    saved = sitk.ReadImage(str(stage / 'draft_labels.nrrd'))
    if not np.array_equal(sitk.GetArrayFromImage(saved), segmentation) or not np.allclose(saved.GetSpacing(), image.GetSpacing()) or not np.allclose(saved.GetOrigin(), image.GetOrigin()) or not np.allclose(saved.GetDirection(), image.GetDirection()):
        raise RuntimeError('Model draft write/read verification failed')
    receipt = {'schema': 'cbct-pretrained-draft/0.2', 'status': 'MODEL_DRAFT', 'case_id': request['case_id'],
               'cohort': '3D_ANNOTATION_ONLY', 'model_id': draft.MODEL_ID, 'model_manifest_sha256': info['model_manifest_sha256'],
               'protocol_sha256': protocol_hash, 'input_file_sha256': input_hash,
               'source_volume_signature': request['source_volume']['signature'],
               'draft_file_sha256': core.digest_file(stage / 'draft_labels.nrrd'),
               'classes': {str(k): v for k, v in draft.LABELS.items()}, 'class_voxels': counts,
               'geometry': request['source_volume'], 'mask_postprocessing': [],
               'model_internal_resampling': 'frozen_checkpoint_plans_with_native_input_grid_output',
               'required_manual_review': list(core.CATEGORIES), 'FDI': 'NOT_PREDICTED', 'D_F': 'NOT_PREDICTED',
               'uncertainty': 'NO_CALIBRATED_UNCERTAINTY_MAP; critical regions require human review',
               'training_admission': False, 'clinical_validation': False,
               'inference_seconds': round(time.monotonic() - started, 3), 'forward_calls': tiles[0],
               'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'runtime': info}
    (stage / 'draft_receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    (stage / 'SHA256SUMS.txt').write_text(''.join(core.digest_file(p) + '  ' + p.name + '\n' for p in sorted(stage.iterdir()) if p.is_file()), encoding='utf-8')
    cancelled(job)
    os.rename(stage, job / 'result')
    (job / 'RUNNING').unlink()
    print(json.dumps({'stage': 'complete', 'status': 'MODEL_DRAFT', 'inference_seconds': receipt['inference_seconds']}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--job', type=Path)
    args = parser.parse_args()
    try:
        if args.preflight:
            print(json.dumps(preflight()), flush=True)
        elif args.job:
            run(args.job)
        else:
            parser.error('Select --preflight or --job')
        return 0
    except Exception as exc:
        payload = {'status': 'CANCELLED' if isinstance(exc, Cancelled) else 'FAILED', 'error': str(exc), 'traceback': traceback.format_exc()}
        if args.job and args.job.is_dir():
            (args.job / 'FAILED.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
        print(json.dumps(payload), flush=True)
        return 130 if isinstance(exc, Cancelled) else 1


if __name__ == '__main__':
    sys.exit(main())
