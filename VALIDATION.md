# Dentra v0.3 validation

Verified against the delivered Python source hashes on 2026-10-20.

| Check | Result |
| --- | --- |
| Core rules and instance/FDI contracts | 41 tests PASS |
| Channel-streamed vs stock native export | 16 fixed synthetic comparisons, exactly equal labels |
| Existing-mask Slicer workflow | 32 checks PASS |
| Aggregate draft bridge and real WSL preflight | 16 checks PASS |
| Instance/FDI Slicer workflow and real import | 16 checks PASS |
| Slicer 5.10.0 module registration | PASS |
| Existing local masks | 6 files, 21 segments; read-only roundtrip PASS |
| Source bytes | 6 segmentations and 241 source DICOM files unchanged |
| Word and PDF guide | 14 pages rendered and visually inspected |

One local CBCT image produced **58 retained instances** and **30 nonempty FDI proposals** in 497.257 seconds of backend execution. 36 instances carry a duplicate, unknown or border-only flag. These counts describe model output, not the number of true teeth or correct FDIs. Every FDI remains clinician-pending. Raw border/core foreground equality was checked before publication and again at import; source provenance survived version export.

The original stock 33-channel native export attempt was interrupted for RAM/swap pressure; no result was published from it. A separate earlier integration attempt failed on an import path before inference. Failed jobs remain local. The replacement calls the same frozen resampler per class and uses first-wins argmax. Synthetic exactness is evidence for that export optimization only, not equivalence to the full published ToothSeg pipeline.

The ToothSeg source snapshot, archive size, MD5, SHA-256, ZIP CRC and six selected model files were verified before loading. Both networks strictly loaded their archived architecture and tensor keys. Dataset tooth names were checked against the FDI mapping. Weights, patch, spacing, TTA and numbering rules were not searched or tuned on local labels.

No Dice, FDI accuracy, D/F sensitivity, diagnostic performance or independent external validation was measured. Expert anatomical acceptance and cross-cohort patient/sequence adjudication remain pending. This tool was not trained; G04 and 2.5D were not used. Training admission and clinical validation remain false even after review completion.

The base research environment still has no nnunetv2, timm 1.0.27 and PyTorch 2.10.0+cu128. Added packages are in the separate dentra_v03 overlay. The actual model run and raw image/mask jobs remain local and are excluded from Git and the code ZIP.

Machine-readable evidence is in validation/RELEASE_VALIDATION_V03.json and the listed test reports. The initial GitHub snapshot omits patient-bearing local paths and old data-bearing audit history. Historical v0.1/v0.2 releases remain separately preserved locally and are not current-source validation.
