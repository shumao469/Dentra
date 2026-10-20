"""Conservative native-grid ToothSeg assembly; no foreground cleanup.

Instance IDs are arbitrary. Semantic classes are mapped through the VERIFIED
dataset.json; FDI suggestions never count as clinician-confirmed labels.
"""
from collections import Counter
import numpy as np

MODEL_ID = 'ToothSeg_14893540_Dentra_native_v1'
ASSEMBLY = ['native_grid_core_components', 'foreground_masked_watershed', 'hard_semantic_vote']


def class_to_fdi(dataset):
    """Accept only explicit permanent FDI names in the archived dataset metadata."""
    from .core import FDI
    positions = ['Central Incisor', 'Lateral Incisor', 'Canine', 'First Premolar',
                 'Second Premolar', 'First Molar', 'Second Molar', 'Third Molar (Wisdom Tooth)']
    names = {quadrant + ' ' + tooth: str(q * 10 + t)
             for q, quadrant in enumerate(['Upper Right', 'Upper Left', 'Lower Left', 'Lower Right'], 1)
             for t, tooth in enumerate(positions, 1)}
    result = {}
    for name, label in dataset['labels'].items():
        if int(label) == 0:
            continue
        # ToothFairy2 released names are inspected and frozen in the manifest.
        text = names.get(str(name).strip(), str(name).strip())
        if text not in FDI:
            raise ValueError('Unverified semantic label name: ' + str(name))
        result[int(label)] = text
    if len(result) != 32 or set(result) != set(range(1, 33)):
        raise ValueError('Expected 32 verified permanent-tooth semantic classes')
    if len(set(result.values())) != 32:
        raise ValueError('Duplicate FDI class definition')
    return result


def assemble(border_core, semantic, mapping):
    """Partition only the native predicted foreground; keep tiny/border-only parts.

    Six-connected cores seed a masked flat watershed. Separate connected
    foreground components without a core get their own retained instance.
    This is NOT the published ToothSeg self-correction pipeline.
    """
    from scipy import ndimage as ndi
    from skimage.segmentation import watershed
    border_core, semantic = np.asarray(border_core), np.asarray(semantic)
    if border_core.ndim != 3 or border_core.shape != semantic.shape:
        raise ValueError('Branch outputs must share the native 3D CBCT grid')
    if not np.issubdtype(border_core.dtype, np.integer) or not np.issubdtype(semantic.dtype, np.integer):
        raise ValueError('Integer branch outputs required')
    if not np.isin(border_core, [0, 1, 2]).all() or not np.isin(semantic, [0, *mapping]).all():
        raise ValueError('Unknown border/core or semantic class')
    foreground = border_core > 0
    if not foreground.any():
        raise ValueError('No tooth foreground was predicted')
    connectivity = ndi.generate_binary_structure(3, 1)
    components, n_components = ndi.label(foreground, connectivity)
    cores, n_cores = ndi.label(border_core == 1, connectivity)
    seeded = set(np.unique(components[cores > 0]).tolist())
    missing = [v for v in range(1, n_components + 1) if v not in seeded]
    # Single marker per border-only component; preserve ALL of its foreground.
    boxes = ndi.find_objects(components)
    for offset, component in enumerate(missing, start=1):
        box = boxes[component - 1]
        local = components[box] == component
        first = tuple(int(x[0]) for x in np.where(local))
        cores[tuple(first[d] + box[d].start for d in range(3))] = n_cores + offset
    del components
    instances = watershed(np.zeros(border_core.shape, np.uint8), cores,
                          connectivity=connectivity, mask=foreground,
                          watershed_line=False).astype(np.uint32)
    del cores
    if not np.array_equal(instances > 0, foreground):
        raise RuntimeError('Foreground conservation failed; no draft may be published')
    # One pass of pair counts avoids scanning the entire CBCT once per tooth.
    active = instances[foreground].astype(np.int64)
    pairs = active * 33 + semantic[foreground].astype(np.int64)
    counts = np.bincount(pairs)
    rows = []
    for instance in np.unique(active):
        votes = np.pad(counts[instance * 33:(instance + 1) * 33], (0, max(0, (instance + 1) * 33 - len(counts))))
        total = int(votes.sum())
        maximum = int(votes.max())
        winners = np.flatnonzero(votes == maximum)
        cls = int(winners[0]) if len(winners) == 1 else 0
        proposal = mapping.get(cls, '')
        flags = []
        if not proposal:
            flags.append('BACKGROUND_OR_TIED_SEMANTICS')
        if int(instance) > n_cores:
            flags.append('BORDER_ONLY_NO_CORE')
        rows.append({'instance_id': int(instance), 'proposed_fdi': proposal,
                     'semantic_class': cls, 'voxels': total,
                     'hard_vote_fraction': maximum / total,
                     'semantic_background_fraction': int(votes[0]) / total,
                     'flags': flags, 'confirmed': False})
    duplicates = Counter(row['proposed_fdi'] for row in rows if row['proposed_fdi'])
    for row in rows:
        if duplicates[row['proposed_fdi']] > 1:
            row['flags'].append('DUPLICATE_FDI_PROPOSAL')
    return instances, rows


def validate_instances(array, receipt):
    array = np.asarray(array)
    if array.ndim != 3 or not np.issubdtype(array.dtype, np.integer) or np.any(array < 0):
        raise ValueError('Expected a nonnegative integer instance map')
    ids, counts = np.unique(array, return_counts=True)
    observed = {int(k): int(v) for k, v in zip(ids, counts) if k}
    rows = receipt.get('instances', [])
    declared = {int(r['instance_id']): int(r['voxels']) for r in rows}
    if not observed or len(declared) != len(rows) or observed != declared:
        raise ValueError('Instance table and voxel map differ')
    from .core import FDI
    for row in rows:
        if row.get('confirmed') is not False or (row['proposed_fdi'] and row['proposed_fdi'] not in FDI):
            raise ValueError('FDI proposals must be valid and remain unconfirmed')
    return {str(k): v for k, v in zip(ids.tolist(), counts.tolist())}
