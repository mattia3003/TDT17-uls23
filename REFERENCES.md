# References and external resources

All external code, data and sources used in the project. Required by the project rules.

## Data and challenge

- M. J. J. de Grauw, E. Th. Scholten, E. J. Smit, M. J. C. M. Rutten, M. Prokop, B. van Ginneken, A. Hering, *The ULS23 challenge: A baseline model and benchmark dataset for 3D universal lesion segmentation in computed tomography*, Medical Image Analysis 102 (2025) 103525. https://doi.org/10.1016/j.media.2025.103525 (preprint: https://arxiv.org/abs/2406.05231)
- F. Isensee et al., *nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation*, Nature Methods 18 (2021) 203–211. The ULS23 baseline and the design of MONAI `DynUNet` are based on it.
- ULS23 challenge site: https://uls23.grand-challenge.org
- ULS23 repository (data description, MONAI version note): https://github.com/MJJdG/ULS23
- ULS23 public training data license: CC BY-NC-SA 4.0. The original source datasets (KiTS21, LIDC-IDRI, LiTS, Medical Segmentation Decathlon, NIH Lymph Nodes) are cited in the dataset README. The ones we use will be added here after Gate 1.

## Software

- MONAI: M. J. Cardoso et al., *MONAI: An open-source framework for deep learning in healthcare*, 2022. https://arxiv.org/abs/2211.02701 (`DynUNet`, `DiceCELoss`, transforms, sliding-window inference)
- PyTorch: https://pytorch.org
- NiBabel, NumPy, SciPy, pandas, Matplotlib, PyYAML (see `requirements.txt` for versions)

## Compute

- NTNU IDUN cluster documentation: https://www.hpc.ntnu.no/idun/
