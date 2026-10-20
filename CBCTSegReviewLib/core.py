"""Pure review/QC rules. Coordinates: RAS mm; arrays: KJI. No mask modification."""
from __future__ import annotations

import hashlib
import datetime
import json
import re
from pathlib import Path

import numpy as np

VERSION = "0.3.0"
CATEGORIES = ("thin_plate", "fenestration", "dehiscence")
ROLES = ("tooth", "alveolar_bone", "jaw_bone", "teeth_group", "nerve_canal", "other")
FDI = {str(q * 10 + t) for q in range(1, 5) for t in range(1, 9)}


def digest_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def nrrd_metadata(path):
    """Header only; distinguish NRRD fields ':' from metadata ':='."""
    result = {}
    with open(path, "rb") as stream:
        total = 0
        for raw in stream:
            total += len(raw)
            if total > 1024 * 1024:
                raise ValueError("Unexpectedly large NRRD header")
            line = raw.decode("utf-8").strip()
            if not line:
                break
            if line.startswith("#") or line.startswith("NRRD"):
                continue
            separator = ":=" if ":=" in line else ":"
            if separator in line:
                key, value = line.split(separator, 1)
                result[key.strip()] = value.strip()
    return result


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(",", ":")).encode()).hexdigest()


def check_affine(matrix):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("Invalid affine")
    if not np.allclose(matrix[3], [0, 0, 0, 1]) or abs(np.linalg.det(matrix[:3, :3])) < 1e-12:
        raise ValueError("Singular/non-affine geometry")
    return matrix


def snapshot(array_kji, ijk_to_ras):
    """Canonical tight bounding box, copied; ignore unused zero padding only."""
    a = np.asarray(array_kji) != 0
    if a.ndim != 3:
        raise ValueError("Expected a 3D binary mask")
    matrix = check_affine(ijk_to_ras).copy()
    if not a.any():
        return {"array": np.zeros((0, 0, 0), dtype=bool), "matrix": matrix}
    bounds = []
    for axis in range(3):
        projections = np.any(a, axis=tuple(i for i in range(3) if i != axis))
        indices = np.flatnonzero(projections)
        bounds.append((int(indices[0]), int(indices[-1]) + 1))
    matrix[:3, 3] = (matrix @ [bounds[2][0], bounds[1][0], bounds[0][0], 1])[:3]
    a = a[tuple(slice(lo, hi) for lo, hi in bounds)].copy()
    return {"array": a, "matrix": matrix}


def fingerprint(snap):
    a = snap["array"]
    geometry = {"shape_kji": list(a.shape), "matrix": np.round(snap["matrix"], 6).tolist()}
    # Empty geometry still participates: changing an empty segment grid is a change.
    return hashlib.sha256(json_hash(geometry).encode() + np.packbits(a).tobytes()).hexdigest()


def delta(before, after):
    """Exact foreground comparison on a common lattice; never interpolate labels."""
    a, b = before["array"], after["array"]
    na, nb = int(a.sum()), int(b.sum())
    if not na or not nb:
        return {"added_voxels": nb, "removed_voxels": na, "comparison": "empty_mask"}
    transform = np.linalg.inv(before["matrix"]) @ after["matrix"]
    offset = transform[:3, 3]
    if not np.allclose(transform[:3, :3], np.eye(3), atol=1e-6) or not np.allclose(offset, np.round(offset), atol=1e-5):
        return {"added_voxels": None, "removed_voxels": None, "comparison": "GRID_CHANGED_NO_RESAMPLING"}
    offset = np.round(offset).astype(int)[::-1]
    lo = np.maximum(0, offset)
    hi = np.minimum(a.shape, np.array(b.shape) + offset)
    intersection = 0
    if np.all(hi > lo):
        aa = a[tuple(slice(int(l), int(h)) for l, h in zip(lo, hi))]
        bb = b[tuple(slice(int(l - o), int(h - o)) for l, h, o in zip(lo, hi, offset))]
        intersection = int(np.count_nonzero(aa & bb))
    return {"added_voxels": nb - intersection, "removed_voxels": na - intersection,
            "comparison": "native_lattice"}


def mask_qc(snap, volume_matrix, volume_shape_kji):
    """Geometry containment, not patient identity, anatomy or diagnosis verification."""
    a = snap["array"]
    matrix = check_affine(snap["matrix"])
    volume_matrix = check_affine(volume_matrix)
    shape = np.asarray(volume_shape_kji, dtype=int)
    if shape.shape != (3,) or (shape <= 0).any():
        raise ValueError("Invalid volume dimensions")
    to_volume = np.linalg.inv(volume_matrix) @ matrix
    count, outside, boundary = int(a.sum()), 0, 0
    # Slice by slice avoids generating a whole-volume foreground point cloud.
    for k in range(a.shape[0]):
        j, i = np.nonzero(a[k])
        if not len(i):
            continue
        points = np.stack((i, j, np.full(len(i), k), np.ones(len(i))))
        coords = (to_volume @ points)[:3].T
        upper = shape[::-1] - 0.5
        outside += int(np.any((coords < -0.50001) | (coords > upper + 1e-5), axis=1).sum())
        boundary += int(np.any((coords < 0.5) | (coords > upper - 1), axis=1).sum())
    return {"voxels": count, "volume_mm3": count * abs(float(np.linalg.det(matrix[:3, :3]))),
            "outside_reference_voxels": outside, "reference_boundary_voxels": boundary,
            "same_axis_sampling_as_volume": bool(np.allclose(to_volume[:3, :3], np.eye(3), atol=1e-5)),
            "ijk_to_ras": matrix.tolist(), "shape_kji": list(a.shape),
            "anatomical_quality": "NOT_AUTOMATICALLY_DETERMINED"}


def mapping_error(mapping):
    role = mapping.get("role")
    if role not in ROLES:
        return "tissue_role_unconfirmed"
    if not str(mapping.get("note", "")).strip() and role == "other":
        return "other_role_requires_note"
    fdi = str(mapping.get("fdi", "")).strip()
    if role == "tooth" and fdi not in FDI:
        return "tooth_requires_explicit_fdi"
    if role != "tooth" and fdi:
        return "fdi_only_for_tooth"
    if not mapping.get("confirmed"):
        return "mapping_unconfirmed"
    return None


def safe_case_id(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", value or ""):
        raise ValueError("Case ID must be a pseudonym: ASCII letters/digits/_/-, 1-64 characters")
    return value


def revision_id(hashes, mappings, regions, volume_hash, segment_names=None):
    return json_hash({"segments": hashes, "mappings": mappings, "regions": regions,
                      "volume_hash": volume_hash, "segment_names": segment_names or {}})


def review_record(revision, reviewer, status, note):
    if not reviewer.strip() or not note.strip():
        raise ValueError("Reviewer ID and explicit review note are required")
    if status not in ("reviewed", "not_applicable", "uncertain"):
        raise ValueError("Invalid review status")
    return {"revision": revision, "reviewer": reviewer.strip(), "status": status, "note": note.strip(),
            "recorded_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}


def release_check(segment_qc, mappings, reviews, regions, revision, binding_confirmed):
    blockers, warnings = [], []
    if not binding_confirmed:
        blockers.append("reference_patient_binding_unconfirmed")
    if not segment_qc:
        blockers.append("no_segments")
    teeth = []
    for sid, qc in segment_qc.items():
        if not qc["voxels"]:
            blockers.append(f"{sid}:empty_segment")
        if qc["outside_reference_voxels"]:
            blockers.append(f"{sid}:outside_reference_volume")
        if qc.get("delta", {}).get("comparison") == "GRID_CHANGED_NO_RESAMPLING":
            blockers.append(f"{sid}:mask_grid_changed")
        if qc["reference_boundary_voxels"]:
            warnings.append(f"{sid}:check_volume_boundary")
        if mappings.get(sid, {}).get("role") in ("alveolar_bone", "jaw_bone") and (qc.get("delta", {}).get("added_voxels") or 0) > 0:
            warnings.append(f"{sid}:bone_added_check_thin_plate_and_true_defect_margins")
        if not qc["same_axis_sampling_as_volume"]:
            warnings.append(f"{sid}:different_sampling_no_automatic_resampling")
        error = mapping_error(mappings.get(sid, {}))
        if error:
            blockers.append(f"{sid}:{error}")
        elif mappings[sid]["role"] == "tooth":
            teeth.append(mappings[sid]["fdi"])
    if len(teeth) != len(set(teeth)):
        blockers.append("duplicate_fdi_mapping")
    if not any(m.get("role") == "alveolar_bone" for m in mappings.values()):
        warnings.append("no_confirmed_alveolar_bone_segment_not_a_full_3D_labelset")
    review_keys = list(CATEGORIES) + ["region:" + r["id"] for r in regions]
    for key in review_keys:
        item = reviews.get(key, {})
        if item.get("revision") != revision or item.get("status") not in ("reviewed", "not_applicable"):
            blockers.append(key + ":review_pending_or_stale")
        elif not item.get("reviewer", "").strip() or not item.get("note", "").strip():
            blockers.append(key + ":missing_review_evidence")
        elif key.startswith("region:") and item["status"] != "reviewed":
            blockers.append(key + ":marked_region_requires_review")
    return {"status": "REVIEW_COMPLETE" if not blockers else "IN_PROGRESS",
            "blockers": blockers, "warnings": warnings, "training_admission": False,
            "clinical_validation": False}


class DraftProvider:
    """Reserved interface: no model or automatic label generation in v0.1."""
    def generate(self, *args, **kwargs):
        raise NotImplementedError("Semi-automatic drafts are not enabled in this release")
