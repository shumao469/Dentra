"""Slicer boundary. Original files and native voxel grids are preserved."""
from __future__ import annotations

import datetime
import hashlib
import json
import os
from pathlib import Path
import uuid

import numpy as np
import slicer
import vtk
from vtk.util.numpy_support import vtk_to_numpy

from . import core


def matrix_array(matrix):
    return np.array([[matrix.GetElement(i, j) for j in range(4)] for i in range(4)])


def require_untransformed(node):
    if node.GetParentTransformNode():
        raise ValueError("Parent transform present. Explicit coordinate audit required; no automatic hardening.")


def volume_info(node):
    require_untransformed(node)
    matrix = vtk.vtkMatrix4x4()
    node.GetIJKToRASMatrix(matrix)
    array = slicer.util.arrayFromVolume(node)
    if array.ndim != 3:
        raise ValueError("Select a scalar 3D CBCT volume")
    affine = matrix_array(matrix)
    extent = node.GetImageData().GetExtent()
    affine[:3, 3] = (affine @ [extent[0], extent[2], extent[4], 1])[:3]
    info = {"shape_kji": list(array.shape), "ijk_to_ras": affine.tolist(),
            "pixel_sha256": hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast("B")).hexdigest(),
            "dtype": str(array.dtype), "coordinate_system": "RAS_mm"}
    info["signature"] = core.json_hash(info)
    return info


def native_snapshots(node):
    require_untransformed(node)
    result = {}
    segmentation = node.GetSegmentation()
    if segmentation.GetSourceRepresentationName() != "Binary labelmap":
        raise ValueError("Existing binary labelmap required; surface-to-label conversion is not automatic")
    for n in range(segmentation.GetNumberOfSegments()):
        sid = segmentation.GetNthSegmentID(n)
        image = slicer.vtkOrientedImageData()
        if not node.GetBinaryLabelmapRepresentation(sid, image):
            raise ValueError("No binary labelmap: " + sid)
        dims = image.GetDimensions()
        matrix = vtk.vtkMatrix4x4()
        image.GetImageToWorldMatrix(matrix)
        affine = matrix_array(matrix)
        extent = image.GetExtent()
        affine[:3, 3] = (affine @ [extent[0], extent[2], extent[4], 1])[:3]
        if np.prod(dims) == 0:
            array = np.zeros((0, 0, 0), dtype=bool)
        else:
            array = vtk_to_numpy(image.GetPointData().GetScalars()).reshape(dims[::-1])
        result[sid] = core.snapshot(array, affine)
    return result


class ReviewSession:
    def __init__(self, segmentation_path, volume, case_id):
        self.case_id = core.safe_case_id(case_id)
        self.original_path = Path(segmentation_path).resolve(strict=True)
        if not self.original_path.name.lower().endswith(".seg.nrrd"):
            raise ValueError("Select an existing .seg.nrrd file")
        self.original_hash = core.digest_file(self.original_path)
        self.original_storage = core.nrrd_metadata(self.original_path)
        self.volume = volume
        self.reference = volume_info(volume)
        self.baseline = None
        self.working = None
        self.markups = None
        try:
            self.baseline = slicer.util.loadSegmentation(str(self.original_path))
            self.baseline.SetName(self.case_id + "_ORIGINAL_DO_NOT_EDIT")
            self.baseline.SetSaveWithScene(False)
            # Never leave the source path as a writable save target in the protected node.
            if self.baseline.GetStorageNode():
                self.baseline.GetStorageNode().SetFileName(None)
            self.before = native_snapshots(self.baseline)
            self.draft_provenance = None
            for sid in self.before:
                segment = self.baseline.GetSegmentation().GetSegment(sid)
                source_kind, provenance = vtk.mutable(''), vtk.mutable('')
                if segment.GetTag('CBCT.SourceKind', source_kind) and str(source_kind) == 'MODEL_DRAFT':
                    if not segment.GetTag('CBCT.DraftReceiptJSON', provenance):
                        raise ValueError('Model draft provenance missing')
                    value = json.loads(str(provenance))
                    if self.draft_provenance is not None and value != self.draft_provenance:
                        raise ValueError('Inconsistent model draft provenance')
                    self.draft_provenance = value
            self.original_names = {sid: self.baseline.GetSegmentation().GetSegment(sid).GetName() for sid in self.before}
            self.working = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", self.case_id + "_WORKING")
            self.working.GetSegmentation().DeepCopy(self.baseline.GetSegmentation())
            self.working.CreateDefaultDisplayNodes()
            self.working.GetDisplayNode().SetOpacity2DFill(0.15)
            self.baseline.GetDisplayNode().SetVisibility(False)
            self.baseline.SetHideFromEditors(True)
            self.markups = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", self.case_id + "_CRITICAL_REGIONS")
            self.markups.CreateDefaultDisplayNodes()
            self.markups.SetLocked(True)
            self.mappings = {sid: {"role": "", "fdi": "", "confirmed": False, "note": ""} for sid in self.before}
            self.reviews = {}
            self.binding_confirmed = False
            self.last_export = None
        except Exception:
            self.close()
            raise

    def close(self):
        for name in ("baseline", "working", "markups"):
            node = getattr(self, name, None)
            if node and node.GetScene():
                slicer.mrmlScene.RemoveNode(node)
            setattr(self, name, None)

    def assert_source_unchanged(self):
        if core.digest_file(self.original_path) != self.original_hash:
            raise ValueError("Original segmentation changed on disk; stop and open a new session")
        if volume_info(self.volume)["signature"] != self.reference["signature"]:
            raise ValueError("Reference image pixels/geometry changed; open a new session")
        baseline = native_snapshots(self.baseline)
        if set(baseline) != set(self.before) or any(core.fingerprint(baseline[s]) != core.fingerprint(self.before[s]) for s in self.before):
            raise ValueError("Protected baseline node changed; open a new session")

    def has_unexported_content(self):
        _, _, regions, revision = self.state()
        if self.last_export:
            saved = json.loads((self.last_export / "review.json").read_text(encoding="utf-8"))
            return revision != saved["revision"] or self.reviews != saved["reviews"]
        current = native_snapshots(self.working)
        return (any(core.fingerprint(current[sid]) != core.fingerprint(self.before[sid]) for sid in self.before)
                or any(value["confirmed"] for value in self.mappings.values()) or bool(regions) or bool(self.reviews)
                or any(self.working.GetSegmentation().GetSegment(sid).GetName() != self.original_names[sid] for sid in self.before))

    def set_mapping(self, sid, role, fdi="", note=""):
        if sid not in self.before:
            raise ValueError("Unknown segment ID")
        value = {"role": role, "fdi": str(fdi).strip(), "confirmed": True, "note": note.strip()}
        error = core.mapping_error(value)
        if error:
            raise ValueError(error)
        if self.draft_provenance:
            if role == 'other' and not note.strip():
                raise ValueError('Explain any manual rejection/reclassification of a model segment')
            from .draft import MODEL_ROLES
            from .toothseg import MODEL_ID as TOOTHSEG
            model_label = vtk.mutable('')
            if self.baseline.GetSegmentation().GetSegment(sid).GetTag('CBCT.ModelLabel', model_label):
                expected = 'tooth' if self.draft_provenance.get('model_id') == TOOTHSEG else MODEL_ROLES[int(str(model_label))]
                if role not in (expected, 'other'):
                    raise ValueError('Preserve aggregate model semantics; do not relabel jaws as alveolar bone or tooth groups as FDI instances')
        self.mappings[sid] = value

    def add_region(self, category, sid, ras, note):
        if category not in core.CATEGORIES or sid not in self.before:
            raise ValueError("Select critical category and existing segment")
        ras = np.asarray(ras, dtype=float)
        if ras.shape != (3,) or not np.isfinite(ras).all() or not note.strip():
            raise ValueError("Finite RAS coordinate and location note required")
        ijk = (np.linalg.inv(np.asarray(self.reference["ijk_to_ras"])) @ np.r_[ras, 1])[:3]
        if np.any(ijk < -0.5) or np.any(ijk > np.asarray(self.reference["shape_kji"])[::-1] - 0.5):
            raise ValueError("Review marker lies outside reference image")
        index = self.markups.AddControlPoint(vtk.vtkVector3d(*ras))
        rid = self.markups.GetNthControlPointID(index)
        self.markups.SetNthControlPointLabel(index, category + ":" + self.original_names[sid])
        self.markups.SetNthControlPointDescription(index, json.dumps({"category": category, "segment_id": sid, "note": note.strip()}))
        return rid

    def regions(self):
        require_untransformed(self.markups)
        regions = []
        for i in range(self.markups.GetNumberOfControlPoints()):
            ras = [0.0, 0.0, 0.0]
            self.markups.GetNthControlPointPositionWorld(i, ras)
            meta = json.loads(self.markups.GetNthControlPointDescription(i))
            if meta.get("category") not in core.CATEGORIES or meta.get("segment_id") not in self.before:
                raise ValueError("Invalid marker metadata")
            regions.append({"id": self.markups.GetNthControlPointID(i), "ras_mm": ras, **meta})
        return regions

    def move_region(self, rid, ras, note):
        region = next((r for r in self.regions() if r["id"] == rid), None)
        if region is None:
            raise ValueError("Unknown review marker")
        ras = np.asarray(ras, dtype=float)
        if ras.shape != (3,) or not np.isfinite(ras).all() or not note.strip():
            raise ValueError("Finite coordinate and note required")
        ijk = (np.linalg.inv(np.asarray(self.reference["ijk_to_ras"])) @ np.r_[ras, 1])[:3]
        if np.any(ijk < -0.5) or np.any(ijk > np.asarray(self.reference["shape_kji"])[::-1] - 0.5):
            raise ValueError("Review marker lies outside reference image")
        for index in range(self.markups.GetNumberOfControlPoints()):
            if self.markups.GetNthControlPointID(index) == rid:
                self.markups.SetNthControlPointPositionWorld(index, *ras)
                self.markups.SetNthControlPointDescription(index, json.dumps({"category": region["category"], "segment_id": region["segment_id"], "note": note.strip()}))
                return

    def remove_region(self, rid):
        for index in range(self.markups.GetNumberOfControlPoints()):
            if self.markups.GetNthControlPointID(index) == rid:
                self.markups.RemoveNthControlPoint(index)
                self.reviews.pop("region:" + rid, None)
                return
        raise ValueError("Unknown review marker")

    def state(self):
        require_untransformed(self.working)
        after = native_snapshots(self.working)
        if set(after) != set(self.before):
            raise ValueError("Segment IDs/count changed. This release refines existing segments only.")
        hashes = {sid: core.fingerprint(snap) for sid, snap in after.items()}
        regions = self.regions()
        names = {sid: self.working.GetSegmentation().GetSegment(sid).GetName() for sid in after}
        revision = core.revision_id(hashes, self.mappings, regions, self.reference["signature"], names)
        return after, hashes, regions, revision

    def review(self, key, reviewer, status, note):
        _, _, regions, revision = self.state()
        if key not in list(core.CATEGORIES) + ["region:" + r["id"] for r in regions]:
            raise ValueError("Unknown review item")
        self.reviews[key] = core.review_record(revision, reviewer, status, note)

    def qc(self):
        self.assert_source_unchanged()
        after, hashes, regions, revision = self.state()
        results = {}
        for sid, snap in after.items():
            qc = core.mask_qc(snap, self.reference["ijk_to_ras"], self.reference["shape_kji"])
            qc["delta"] = core.delta(self.before[sid], snap)
            qc["original_name"] = self.original_names[sid]
            qc["current_name"] = self.working.GetSegmentation().GetSegment(sid).GetName()
            qc["mask_sha256"] = hashes[sid]
            qc["baseline_sha256"] = core.fingerprint(self.before[sid])
            results[sid] = qc
        gate = core.release_check(results, self.mappings, self.reviews, regions, revision, self.binding_confirmed)
        return {"schema": "cbct-seg-review/0.2", "version": core.VERSION, "case_id": self.case_id,
                "source_sha256": self.original_hash, "source_filename": self.original_path.name,
                "original_storage_metadata": {k: v for k, v in self.original_storage.items()
                                              if k.startswith("Segment") and any(k.endswith(s) for s in ("_ID", "_Name", "_LabelValue", "_Layer"))},
                "reference": self.reference, "revision": revision, "segments": results,
                "mappings": self.mappings, "critical_regions": regions, "reviews": self.reviews,
                "gate": gate, "review_scope": "mask refinement only; no diagnostic accuracy claim",
                "source_kind": "MODEL_DRAFT" if self.draft_provenance else "EXISTING_ANNOTATION",
                "draft_provenance": self.draft_provenance,
                "smoothing_applied_by_module": False, "automatic_mask_generation": bool(self.draft_provenance),
                "slicer_version": slicer.app.applicationVersion}

    def export(self, output_root):
        """New version directory only; read-back mask verification before publication."""
        report = self.qc()
        root = Path(output_root).resolve()
        root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        token = uuid.uuid4().hex[:12]
        final = root / f"{self.case_id}_{stamp}_{token}"
        stage = root / (".partial_" + final.name)
        stage.mkdir(exist_ok=False)
        try:
            segpath = stage / "refined.seg.nrrd"
            if not slicer.util.saveNode(self.working, str(segpath)):
                raise RuntimeError("Segmentation write failed")
            readback = slicer.util.loadSegmentation(str(segpath))
            try:
                loaded = native_snapshots(readback)
                if set(loaded) != set(report["segments"]) or any(core.fingerprint(s) != report["segments"][sid]["mask_sha256"] for sid, s in loaded.items()):
                    raise RuntimeError("Saved segmentation round-trip verification failed")
            finally:
                slicer.mrmlScene.RemoveNode(readback)
            if not slicer.util.saveNode(self.markups, str(stage / "critical_regions.mrk.json")):
                raise RuntimeError("Markups write failed")
            report["export_utc"] = stamp
            report["parent_export"] = str(self.last_export.name) if self.last_export else None
            (stage / "review.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
            (stage / "SHA256SUMS.txt").write_text("".join(f"{core.digest_file(p)}  {p.name}\n" for p in sorted(stage.iterdir()) if p.is_file()), encoding="utf-8")
            self.assert_source_unchanged()
            os.rename(stage, final)
            # saveNode sets the storage filename: update it after moving the directory.
            self.working.GetStorageNode().SetFileName(str(final / "refined.seg.nrrd"))
            self.markups.GetStorageNode().SetFileName(str(final / "critical_regions.mrk.json"))
            self.last_export = final
            return final, report
        except Exception as exc:
            # Keep partial evidence for inspection; do not publish as a complete version.
            (stage / "EXPORT_FAILED.txt").write_text(str(exc), encoding="utf-8")
            raise

    def resume(self, version_directory):
        """Resume only against the original file and exact reference image signature."""
        directory = Path(version_directory).resolve(strict=True)
        checks = directory / "SHA256SUMS.txt"
        checked = set()
        for line in checks.read_text(encoding="utf-8").splitlines():
            expected, name = line.split("  ", 1)
            if Path(name).name != name or name in checked or core.digest_file(directory / name) != expected:
                raise ValueError("Version checksum failed")
            checked.add(name)
        if checked != {"refined.seg.nrrd", "critical_regions.mrk.json", "review.json"}:
            raise ValueError("Incomplete version checksum manifest")
        manifest = json.loads((directory / "review.json").read_text(encoding="utf-8"))
        if manifest["source_sha256"] != self.original_hash or manifest["reference"]["signature"] != self.reference["signature"] or manifest["case_id"] != self.case_id:
            raise ValueError("Original segmentation/reference/case mismatch")
        segmentation = markers = None
        try:
            segmentation = slicer.util.loadSegmentation(str(directory / "refined.seg.nrrd"))
            markers = slicer.util.loadMarkups(str(directory / "critical_regions.mrk.json"))
            loaded = native_snapshots(segmentation)
            if set(loaded) != set(self.before) or any(core.fingerprint(s) != manifest["segments"][sid]["mask_sha256"] for sid, s in loaded.items()):
                raise ValueError("Version mask fingerprint mismatch")
            slicer.mrmlScene.RemoveNode(self.working)
            slicer.mrmlScene.RemoveNode(self.markups)
            self.working, self.markups = segmentation, markers
            segmentation = markers = None
            self.working.SetName(self.case_id + "_WORKING")
            self.markups.SetLocked(True)
            self.mappings = manifest["mappings"]
            self.reviews = manifest["reviews"]
            self.binding_confirmed = False  # operator must reconfirm patient-image binding after reopening
            self.last_export = directory
        finally:
            for node in (segmentation, markers):
                if node:
                    slicer.mrmlScene.RemoveNode(node)
