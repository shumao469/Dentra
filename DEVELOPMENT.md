# Dentra development and merge rules

Work remains on an independent 3D annotation branch. The 2.5D pipeline, its five fixed folds and 392-tooth denominator are outside this repository. G04 is excluded, collapsed 959-tooth pseudolabels remain disabled, and no training is authorized.

## Architecture

`CBCTSegReview.py` retains its Slicer registration identifier for compatibility; its displayed title is Dentra. The UI selects a provider and runs a local WSL process asynchronously. PyTorch is not installed inside Slicer. `draft_bridge.py` exports the selected image, binds it to a signed request, verifies the provider protocol and weight manifest, then imports native-grid masks into a protected baseline and an editable copy.

`backend/infer.py` retains the five aggregate DentalSegmentator classes. `backend/toothseg_infer.py` runs the released semantic and border/core branches sequentially, strictly loads checkpoint tensors into the archived plans architecture, and records runtime settings. `toothseg.py` preserves the raw native border/core foreground while partitioning instances, then records hard-vote FDI proposals. Model normalization and resampling are recorded; no inference-mask cleaning is added. Do not claim equivalence to the published ToothSeg pipeline.

## Preservation and review

Keep raw image, baseline, instance IDs, semantic mapping, raw FDI suggestions and clinician corrections distinct. Each job uses a new directory. Partial results and failed attempts remain available and cannot be overwritten; only complete results are imported. Native mask readback, geometry and hashes are checked. Role/FDI confirmation and all critical-region reviews remain pending until a human acts. Content changes invalidate reviews. REVIEW_COMPLETE does not grant training admission or clinical validity.

Fine alveolar-bone truth and D/F diagnoses are not generated. Independent thin-plate, fenestration and dehiscence review remains mandatory. Manual segment addition/split/merge is outside the current fixed-ID review session; design a provenance-preserving protocol before adding it. Future two-stage diagnosis must use patient-fold OOF segmentation, not exclusively expert masks, but no such training runs here.

## Required checks

- Core: `python tools/run_core_tests.py`.
- Existing-mask review: `tests/slicer_integration.py` inside Slicer.
- Aggregate draft and asynchronous dispatch: `tests/slicer_draft_integration.py`.
- Instance/FDI and optional local actual-inference import: `tests/slicer_toothseg_integration.py`. Set DENTRA_PILOT_JOB only to a separately approved local job; never commit its image or job path.
- Registration: `tests/slicer_registration.py`, without --no-main-window.
- Real-mask preservation: `tests/slicer_existing_data_audit.py`, local approved data only.

Rerun the relevant checks after code changes; compare tested_source_sha256 with current code. Keep synthetic results separate from actual network execution, and engineering checks separate from anatomical accuracy. Resource failures do not authorize changing model spacing, patch, weights or enlarging searches.

## Publishing and merging

Only code, fixed protocols, manifests, deidentified validation and guides may enter Git. No patient data, original logs, jobs, model binaries, credentials or machine configuration. Publish an explicit allowlist, verify the staged tree, push normally without force, then compare the remote commit/tree. The initial GitHub snapshot intentionally omits local-only historical audit paths and data-bearing history; original local releases remain preserved.

Use a feature branch and reviewable pull request for future changes. State the concrete behavior, algorithm/provenance differences and relevant verification. Do not merge changes into the 2.5D pipeline implicitly. Check third-party model attribution and distinguish dataset license metadata from pretrained weight record terms. CMake/Extension Manager packaging is unverified; supported loading uses the complete directory and --additional-module-paths.
