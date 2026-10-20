# Third party components

Dentra is an independent research integration. It is not an official ToothSeg, DentalSegmentator or 3D Slicer release.

- ToothSeg: MIC-DKFZ and contributors, Apache-2.0 source, https://github.com/MIC-DKFZ/ToothSeg . Pretrained ToothFairy2 checkpoints are separately distributed under CC-BY-4.0 at https://zenodo.org/records/14893540 . Please cite the paper and record recommended by those authors when using the weights. Dentra changes the inference assembly and numbering procedure; published pipeline accuracy does not transfer to this variant.
- DentalSegmentator: https://github.com/gaudot/SlicerDentalSegmentator ; pretrained v100 record https://zenodo.org/records/10829675 under CC-BY-4.0. Its five aggregate model classes are retained without conversion into tooth instances or fine alveolar-bone truth.
- nnU-Net: https://github.com/MIC-DKFZ/nnUNet , Apache-2.0. Follow its publication citation requirements when publishing results.
- 3D Slicer: https://www.slicer.org , distributed separately under its own license. Slicer is required and is not bundled.
- PyTorch, NumPy, SciPy, scikit-image, SimpleITK and dynamic-network-architectures are separate dependencies with their respective licenses. No dependency binaries or model weights are included here.

No upstream source modules are vendored in this package. Model records and official documentation remain the authority for their terms and citations. This repository does not introduce a clinical approval or a data-sharing license.

The archived ToothFairy2 semantic dataset metadata declares CC-BY-SA 4.0 for the dataset, separately from the pretrained weight record's CC-BY-4.0 declaration. No dataset is redistributed. Keep these declarations distinct when reviewing later reuse or redistribution.
