# Dentra

Local CBCT drafts and expert segmentation review in 3D Slicer.

Dentra creates a protected baseline and an editable copy, records clinician corrections and critical-region reviews, and exports versioned native-grid segmentations with provenance and SHA-256 checksums. This is a research annotation tool; drafts are never automatically admitted to training or treated as clinical truth.

## Draft providers

| Provider | Output | Human confirmation |
| --- | --- | --- |
| ToothSeg branches with Dentra assembly | Separate tooth instances and permanent FDI proposals | Every instance and FDI; duplicates and unknowns remain visible |
| DentalSegmentator v100 | Upper skull/maxilla, mandible, upper/lower teeth groups, mandibular canal | Aggregate anatomy; not individual FDI teeth or fine alveolar-bone truth |
| Existing expert `.seg.nrrd` | Binary segments on their existing grid | Image identity, tissue role, FDI when applicable and anatomical corrections |

The ToothSeg mode partitions the native-grid predicted foreground and uses hard semantic voting. Tiny and border-only components remain; duplicate FDI proposals are never merged. This differs from the published ToothSeg self-correction pipeline; its published accuracy is not claimed for this variant. See [the model and postprocessing audit](TOOTHSEG_AUDIT.md).

## Start locally

1. Install 3D Slicer 5.10.0 on Windows and use a compatible CUDA PyTorch environment in WSL.
2. Copy `config/local_backend.example.json` to `config/local_backend.json`; set the distribution, isolated venv Python and model cache paths. Machine configuration is not in Git.
3. From the repository directory in WSL, run the following with your compatible PyTorch Python. The installer creates the configured isolated overlay venv and verifies the separate public model downloads.

   ```bash
   python tools/install_backend.py --provider toothseg
   python tools/install_backend.py --provider dental
   ```

4. Run `Start_Dentra.cmd` on Windows. For another Slicer location use `Launch_Slicer_Review.ps1 -SlicerPath 'C:/path/Slicer.exe'`.
5. Load the CBCT, enter an anonymous case ID, choose a provider, check deployment, confirm input scope and generate. Existing annotations can be opened directly.
6. Check image binding, refine the copy, confirm or correct FDI proposals, complete separate critical-region reviews, run QC and export a new version.

The tested setup uses GPU 1, an RTX PRO 5000 Blackwell with 48 GB VRAM. The frozen ToothSeg preflight requires 36 GiB free VRAM and admits at most 125 million input voxels. CPU fallback and automatic resolution/patch changes are not implemented. The system-site-packages overlay inherits base PyTorch read-only; it is not a fully independent runtime copy. Whole-directory direct module loading is tested; CMake/Extension Manager distribution is not.

## Review and preservation

- No automatic smoothing, morphological closing, hole filling, foreground expansion or small-component deletion.
- Thin cortical plate, fenestration and dehiscence require separate image-based clinician review. ToothSeg predicts teeth, not alveolar bone or D/F diagnosis. Its training conversion filled tooth reference holes; preserving the neural output does not guarantee preservation of true defects.
- Instance IDs, raw pixel values, semantic classes, model FDI proposals and clinician-confirmed FDIs remain distinct. Hard-vote fractions are not calibrated confidence.
- Review sessions keep a fixed set of segment IDs. A provenance-preserving split/merge or missing-instance creation workflow is future work; ambiguous cases stay pending.
- Content changes invalidate prior review signatures. REVIEW_COMPLETE only describes completed review records; training admission and clinical validation remain false.
- 2.5D code, fixed five-fold weights and the 392-tooth denominator remain isolated. G04 is excluded from import, tuning and formal evaluation. Collapsed 959-tooth pseudolabels remain disabled. Folder names alone do not prove patient identity or cohort independence.

## Documentation and verification

- [中文详细操作指南](OPERATION_GUIDE.md)
- [Word guide](docs/Dentra_Operation_Guide_v03.docx) · [PDF guide](docs/Dentra_Operation_Guide_v03.pdf)
- [Validation evidence and limits](VALIDATION.md)
- [Development and merge rules](DEVELOPMENT.md) · [Codex handoff](CODEX_START_PROMPT.txt)
- [Data contract](data_contract.json) · [Model attribution](THIRD_PARTY_NOTICES.md)

Core tests do not require Slicer, CUDA or model weights:

```bash
python -m pip install numpy scipy scikit-image
python tools/run_core_tests.py
```

Slicer tests cover native-mask preservation, provenance, FDI correction, duplicate blocking, pending review gates, version restore and asynchronous WSL dispatch. Local image smoke tests check engineering execution only. No tooth/defect accuracy or independent external clinical performance has been established.

Patient images, annotations, job directories, weights and machine configuration are excluded from Git. Download weights separately and retain their attribution and license notices. This is an independent integration, not an official ToothSeg, DentalSegmentator or Slicer release.
