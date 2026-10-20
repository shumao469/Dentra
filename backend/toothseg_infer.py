"""Pinned dual-branch ToothSeg weights with Dentra native-grid assembly.

No training, inference-mask cleanup, tooth suppression, arch optimization or
automatic human confirmation. This is an engineering variant of ToothSeg.
"""
from pathlib import Path
import argparse
import datetime
import gc
import hashlib
import importlib.metadata
import json
import os
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from CBCTSegReviewLib import core, draft, toothseg
from infer import Cancelled, cancelled, configuration
from native_export import native_argmax


def model_files():
    config = configuration()
    manifest = json.loads((ROOT / 'config/toothseg_manifest.json').read_text(encoding='utf-8'))
    model_root = Path(config['toothseg_model_root']).resolve(strict=True)
    for name, expected in manifest['files_sha256'].items():
        path = (model_root / name).resolve(strict=True)
        if not path.is_relative_to(model_root) or core.digest_file(path) != expected:
            raise ValueError('ToothSeg model file checksum failed: ' + name)
    return model_root, manifest


def preflight():
    import torch
    protocol, protocol_hash = draft.load_protocol(ROOT, toothseg.MODEL_ID)
    model, manifest = model_files()
    if importlib.metadata.version('nnunetv2') != '2.8.1':
        raise ValueError('Use nnunetv2==2.8.1')
    gpu = int(protocol['gpu'].split(':')[1])
    if not torch.cuda.is_available() or gpu >= torch.cuda.device_count():
        raise ValueError('Configured CUDA GPU unavailable')
    free, total = torch.cuda.mem_get_info(gpu)
    if free < protocol['minimum_free_gpu_gib'] * 1024**3:
        raise ValueError('Insufficient free GPU memory for the frozen ToothSeg protocol')
    return {'status': 'READY', 'python': sys.executable, 'torch': torch.__version__,
            'nnunetv2': importlib.metadata.version('nnunetv2'), 'gpu': gpu,
            'gpu_name': torch.cuda.get_device_name(gpu), 'gpu_free_bytes': free,
            'gpu_total_bytes': total, 'model_id': toothseg.MODEL_ID,
            'model_manifest_sha256': core.digest_file(ROOT / 'config/toothseg_manifest.json'),
            'protocol_sha256': protocol_hash, 'weights_verified': True,
            'training': False, 'postprocessing': []}


def predict_branch(model, branch, array, spacing, protocol, job):
    import numpy as np
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
    from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
    folder = model / branch['folder']
    dataset = json.loads((folder / 'dataset.json').read_text())
    plans = PlansManager(json.loads((folder / 'plans.json').read_text()))
    # The archived custom trainer is absent from nnU-Net 2.8.1. Instantiate the
    # EXACT archived plans architecture; strict state loading rejects mismatches.
    # No trainer is instantiated and no package/source/weight file is modified.
    checkpoint = torch.load(folder / branch['checkpoint'], map_location='cpu', weights_only=False)
    if checkpoint['trainer_name'] != branch['trainer_name']:
        raise ValueError('Unexpected archived trainer')
    manager = plans.get_configuration(checkpoint['init_args']['configuration'])
    if checkpoint['init_args']['configuration'] != branch['configuration']:
        raise ValueError('Checkpoint/configuration mismatch')
    network = get_network_from_plans(manager.network_arch_class_name,
                                    manager.network_arch_init_kwargs,
                                    manager.network_arch_init_kwargs_req_import,
                                    determine_num_input_channels(plans, manager, dataset),
                                    plans.get_label_manager(dataset).num_segmentation_heads,
                                    allow_init=False, deep_supervision=False)
    network.load_state_dict(checkpoint['network_weights'], strict=True)
    predictor = nnUNetPredictor(tile_step_size=protocol['tile_step_size'], use_gaussian=True,
                                use_mirroring=protocol['use_checkpoint_mirroring_axes'],
                                perform_everything_on_device=True, device=torch.device(protocol['gpu']),
                                verbose=False, verbose_preprocessing=False, allow_tqdm=False)
    predictor.manual_initialization(network, plans, manager, [checkpoint['network_weights']],
                                    dataset, checkpoint['trainer_name'], checkpoint['inference_allowed_mirroring_axes'])
    calls = [0]
    def progress(*args):
        cancelled(job)
        calls[0] += 1
        if calls[0] == 1 or calls[0] % 100 == 0:
            print(json.dumps({'stage': branch['name'], 'forward_calls': calls[0]}), flush=True)
    hook = network.register_forward_pre_hook(progress)
    runtime = {'trainer_name': checkpoint['trainer_name'], 'configuration': branch['configuration'],
               'architecture': manager.network_arch_class_name, 'patch_size': manager.patch_size,
               'spacing': manager.spacing, 'strict_state_dict': True,
               'mirroring_axes': checkpoint['inference_allowed_mirroring_axes'],
               'network_loader': 'archived_plans_strict_state_manual_initialization'}
    try:
        with torch.inference_mode():
            from nnunetv2.inference.data_iterators import PreprocessAdapterFromNpy
            preprocessed = next(PreprocessAdapterFromNpy([array[None].astype(np.float32)], [None],
                        [{'spacing': list(spacing)[::-1]}], [None], plans, dataset, manager,
                        num_threads_in_multithreaded=1, verbose=False))
            logits = predictor.predict_logits_from_preprocessed_data(preprocessed['data']).cpu()
            del preprocessed['data']
            print(json.dumps({'stage': branch['name'] + '_native_channel_export'}), flush=True)
            output = native_argmax(logits, plans, manager, predictor.label_manager,
                                   preprocessed['data_properties'], lambda: cancelled(job))
            del logits, preprocessed
    finally:
        hook.remove()
    result = output[0] if isinstance(output, tuple) else output
    runtime['forward_calls'] = calls[0]
    del predictor, network, checkpoint, output
    gc.collect()
    torch.cuda.empty_cache()
    return result, dataset, runtime


def run(job):
    import numpy as np
    import SimpleITK as sitk
    import torch
    job = job.resolve(strict=True)
    if (job / 'result').exists() or (job / 'FAILED.json').exists():
        raise ValueError('Use a fresh job; prior attempts are never overwritten')
    with (job / 'RUNNING').open('x') as stream:
        stream.write(str(os.getpid()))
    started = time.monotonic()
    request = json.loads((job / 'job.json').read_text(encoding='utf-8'))
    protocol, protocol_hash = draft.load_protocol(ROOT, toothseg.MODEL_ID)
    draft.check_request(request, protocol_hash)
    if request['model_id'] != toothseg.MODEL_ID:
        raise ValueError('Wrong backend for this provider')
    info = preflight()
    cancelled(job)
    input_path = job / 'input.nrrd'
    if core.digest_file(input_path) != request['input_file_sha256']:
        raise ValueError('Input file changed')
    image = sitk.ReadImage(str(input_path))
    array = sitk.GetArrayFromImage(image)
    if array.ndim != 3 or array.size > protocol['maximum_input_voxels'] or not np.isfinite(array).all():
        raise ValueError('Invalid image or frozen resource limit exceeded')
    matrix = np.eye(4)
    matrix[:3, :3] = np.diag([-1, -1, 1]) @ np.asarray(image.GetDirection()).reshape(3, 3) @ np.diag(image.GetSpacing())
    matrix[:3, 3] = np.diag([-1, -1, 1]) @ image.GetOrigin()
    draft.check_geometry(array.shape, matrix, request['source_volume'])
    if hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast('B')).hexdigest() != request['source_volume']['pixel_sha256']:
        raise ValueError('Input pixels differ from selected Slicer volume')
    os.environ['nnUNet_compile'] = 'false'
    for key in ('raw', 'preprocessed', 'results'):
        os.environ['nnUNet_' + key] = str(job / ('unused_' + key))
    torch.set_num_threads(protocol['cpu_threads'])
    torch.cuda.set_device(info['gpu'])
    model, manifest = model_files()
    stage = job / '.partial_result'
    stage.mkdir(exist_ok=False)
    def save(name, labels):
        draft.check_geometry(labels.shape, matrix, request['source_volume'])
        output = sitk.GetImageFromArray(labels)
        output.CopyInformation(image)
        sitk.WriteImage(output, str(stage / name), True)
        saved = sitk.ReadImage(str(stage / name))
        if not np.array_equal(sitk.GetArrayFromImage(saved), labels) or saved.GetDirection() != image.GetDirection() or saved.GetOrigin() != image.GetOrigin() or saved.GetSpacing() != image.GetSpacing():
            raise RuntimeError('Native geometry/pixels did not survive file roundtrip')
    branch_runtime = {}
    print(json.dumps({'stage': 'semantic', 'shape_kji': list(array.shape)}), flush=True)
    semantic, dataset, branch_runtime['semantic'] = predict_branch(model, manifest['branches']['semantic'], array, image.GetSpacing(), protocol, job)
    mapping = toothseg.class_to_fdi(dataset)
    if {str(k): v for k, v in mapping.items()} != manifest['semantic_class_to_fdi']:
        raise ValueError('Semantic FDI mapping differs from frozen manifest')
    save('raw_semantic_labels.nrrd', semantic.astype(np.uint8))
    cancelled(job)
    print(json.dumps({'stage': 'border_core'}), flush=True)
    border_core, _, branch_runtime['border_core'] = predict_branch(model, manifest['branches']['border_core'], array, image.GetSpacing(), protocol, job)
    save('raw_border_core.nrrd', border_core.astype(np.uint8))
    cancelled(job)
    print(json.dumps({'stage': 'conservative_instance_assembly'}), flush=True)
    instances, rows = toothseg.assemble(border_core, semantic, mapping)
    save('draft_labels.nrrd', instances)
    receipt = {'schema': 'dentra-toothseg-native/0.3', 'status': 'MODEL_DRAFT',
               'model_id': toothseg.MODEL_ID, 'case_id': request['case_id'], 'cohort': '3D_ANNOTATION_ONLY',
               'protocol_sha256': protocol_hash, 'model_manifest_sha256': info['model_manifest_sha256'],
               'input_file_sha256': request['input_file_sha256'], 'source_volume_signature': request['source_volume']['signature'],
               'draft_file_sha256': core.digest_file(stage / 'draft_labels.nrrd'),
               'raw_semantic_sha256': core.digest_file(stage / 'raw_semantic_labels.nrrd'),
               'raw_border_core_sha256': core.digest_file(stage / 'raw_border_core.nrrd'),
               'geometry': request['source_volume'], 'instances': rows, 'mask_postprocessing': [],
               'instance_assembly': toothseg.ASSEMBLY, 'native_foreground_preserved': True,
               'model_internal_resampling': 'archived_plans_normalization_and_logits_resampling_to_native_grid',
               'native_export': 'channel_streamed_frozen_resampler_first_wins_argmax',
               'FDI': 'UNCONFIRMED_HARD_VOTE_PROPOSALS', 'D_F': 'NOT_PREDICTED',
               'uncertainty': 'hard_vote_fraction_is_not_calibrated_confidence',
               'required_manual_review': list(core.CATEGORIES), 'training_admission': False,
               'clinical_validation': False, 'published_toothseg_pipeline_equivalence': False,
               'inference_seconds': round(time.monotonic() - started, 3), 'runtime': info,
               'branches': branch_runtime, 'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    toothseg.validate_instances(instances, receipt)
    draft.validate_receipt(receipt, request, request['input_file_sha256'], receipt['draft_file_sha256'])
    (stage / 'draft_receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    (stage / 'SHA256SUMS.txt').write_text(''.join(core.digest_file(p) + '  ' + p.name + '\n' for p in sorted(stage.iterdir()) if p.is_file()), encoding='utf-8')
    cancelled(job)
    stage.rename(job / 'result')
    (job / 'RUNNING').unlink()
    print(json.dumps({'stage': 'complete', 'instances': len(rows), 'seconds': receipt['inference_seconds']}), flush=True)


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
        payload = {'status': 'CANCELLED' if isinstance(exc, Cancelled) else 'FAILED',
                   'error': str(exc), 'traceback': traceback.format_exc()}
        if args.job and args.job.is_dir():
            (args.job / 'FAILED.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
        print(json.dumps(payload), flush=True)
        return 130 if isinstance(exc, Cancelled) else 1


if __name__ == '__main__':
    sys.exit(main())
