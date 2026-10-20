# ToothSeg pretrained model and postprocessing audit

Frozen on 2026-10-20 before local image validation. No training or parameter search.

## Sources

- Official source: https://github.com/MIC-DKFZ/ToothSeg/tree/b29d1017fa124f89645fa56f98649e3f3f43bdb0
- Pretrained archive: https://zenodo.org/records/14893540
- `ToothSeg.zip`: 920,019,129 bytes; published MD5 `5d8dd061cce9529943567aeba3271143`.
- ToothSeg source is Apache-2.0; published weights are CC-BY-4.0. Weights are downloaded separately and are not committed to this repository.
- The archived semantic `dataset.json` separately declares its dataset license as CC-BY-SA 4.0. Dataset and weight record declarations are distinct; no dataset is redistributed here.
- The semantic and border/core checkpoints are the released ToothFairy2 models, fold 5. This fold has no relationship to the separate 2.5D project's five folds.

## Original behavior and the selected Dentra variant

The upstream border/core converter defaults to a 16 mm3 small-center threshold and a 16 mm3 minimum instance size. It can demote small centers, remove small instances and reassign disconnected fragments. The general inference script then uses probability-based self-correction and arch constraints; the ToothFairy2 script uses a majority assignment variant that can also drop small or background-only instances. Numbered output may merge multiple instances with the same label. These actions are inappropriate as an unreviewed mask-cleaning default in this annotation workflow.

Dentra uses the two released branches, frozen checkpoint normalization and resampling, then partitions the native-grid border/core foreground with six-connected core markers and a foreground-masked flat watershed. Every foreground component without a core receives a retained instance. There is no size cutoff, foreground expansion, hole filling, smoothing, fragment deletion, or merge by FDI. An exact foreground-equality assertion runs before publication and again at Slicer import. Raw semantic and border/core maps remain in the local job with hashes.

FDI proposals use a hard semantic class vote within each instance, including the background class. Background winners and ties remain unnumbered; duplicate proposals remain separate and are flagged. The vote fraction is descriptive agreement, not calibrated confidence. Human role/FDI confirmation is always pending. Duplicate confirmed FDIs block review completion.

This is **ToothSeg pretrained weights with Dentra conservative native-grid assembly**, not a reproduction of the published full self-correction pipeline. Native-grid partitioning and hard-vote numbering can change instance boundaries, counts and numbering. No published ToothSeg accuracy is claimed for this variant.

## Anatomical limits

The upstream training-data conversion explicitly calls `binary_fill_holes` on tooth reference masks. Dentra does not run that conversion, does not train, and never applies it to local masks. However, the released weights may reflect this training convention. No inference cleanup does **not** guarantee that the neural prediction preserves every real anatomical defect.

ToothSeg predicts teeth, not alveolar bone or D/F diagnoses. Thin cortical plate, fenestration and dehiscence require separate image-based clinician review. DentalSegmentator aggregate jaw/teeth masks remain separate jobs; they are never silently converted into fine alveolar-bone ground truth or overwritten by tooth instances.

## Runtime compatibility

The released custom semantic trainer is absent from nnU-Net 2.8.1. Dentra constructs the archived architecture from the frozen plans and loads all checkpoint tensor keys strictly, preserving the checkpoint mirroring axes. It does not alias trainers, edit installed packages, or rewrite checkpoints. Strict tensor matching checks compatibility, not numerical equivalence to the original software version. Actual runtime architecture, patch, spacing and axes are recorded per branch.

GPU 1, tile step 0.5, Gaussian tile weighting and checkpoint mirroring are frozen. The protocol requires 36 GiB free GPU memory and at most 125 million input voxels. There is no automatic model resolution/patch reduction. Failures retain partial jobs and cannot be overwritten.

## Evidence boundary

The first implementation used stock simultaneous full-channel native-logit export and exhausted practical RAM headroom on the 31 GiB WSL instance. That attempt was interrupted and retained without a completed result. Dentra now calls the same frozen resampler independently per class and applies first-wins argmax incrementally. This avoids allocating all 33 native-grid float channels together. Sixteen fixed synthetic comparisons against stock nnU-Net export (both branches, float16/float32, up/down/same-size and ties, with crop restoration) produced exactly equal native labels. Weights, patch, spacing, normalization, TTA and numbering were not changed. This numerical export check does not establish whole-pipeline equivalence to upstream ToothSeg.

Synthetic tests verify hole/fragment retention, duplicate/unknown FDI handling, source integrity and pending manual gates. A local image smoke checks engineering execution, not Dice, FDI accuracy, sensitivity or clinical validity. Patient/sequence adjudication and cross-cohort identity deduplication remain pending. G04 and 2.5D files, models and evaluations are excluded from this workflow.
