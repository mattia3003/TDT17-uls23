# References and external resources

All external code, data and sources used in the project. Required by the project rules.

## Data and challenge

- M. J. J. de Grauw, E. Th. Scholten, E. J. Smit, M. J. C. M. Rutten, M. Prokop, B. van Ginneken, A. Hering, *The ULS23 challenge: A baseline model and benchmark dataset for 3D universal lesion segmentation in computed tomography*, Medical Image Analysis 102 (2025) 103525. https://doi.org/10.1016/j.media.2025.103525 (preprint: https://arxiv.org/abs/2406.05231)
- F. Isensee et al., *nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation*, Nature Methods 18 (2021) 203–211. The ULS23 baseline and the design of MONAI `DynUNet` are based on it.
- ULS23 challenge site: https://uls23.grand-challenge.org
- ULS23 repository (data description, MONAI version note): https://github.com/MJJdG/ULS23
- ULS23 public training data license: CC BY-NC-SA 4.0. We use only the three ULS23 novel-annotation datasets (DeepLesion3D, Radboudumc_Bone, Radboudumc_Pancreas). The processed public datasets (KiTS21, LIDC-IDRI, LiTS, Medical Segmentation Decathlon, NIH Lymph Nodes) are not on IDUN and are not used.
- DeepLesion, the source of the DeepLesion3D VOIs: K. Yan, X. Wang, L. Lu, R. M. Summers, *DeepLesion: automated mining of large-scale lesion annotations and universal lesion detection with deep learning*, Journal of Medical Imaging 5(3) (2018) 036501. https://doi.org/10.1117/1.JMI.5.3.036501
- Official ULS23 train/test split: `baseline_model/train_test_split.json` in https://github.com/DIAGNijmegen/ULS23, commit `f45a17259bdaaa87dc1a0aaf2f7004c292b73fc7`, SHA-256 `4d781d5c1f5fae991295b41f5d4cc2d3a27cbd781071fae97f17ee0a6d135c0a`. We use its test cases for the two training datasets (D15). The file is downloaded and not redistributed, because the repo shows no license.

## Software

- MONAI: M. J. Cardoso et al., *MONAI: An open-source framework for deep learning in healthcare*, 2022. https://arxiv.org/abs/2211.02701 (`DynUNet`, `DiceCELoss`, transforms, sliding-window inference)
- PyTorch: https://pytorch.org
- NiBabel, NumPy, SciPy, pandas, Matplotlib, PyYAML (see `requirements.txt` for versions)

## Figures

- The categorical colors in the figures are the first slots of the default palette in Claude Code's `dataviz` skill. They were checked for colorblind separation with its validator script.

## Tools

- Claude Code (Anthropic) was used as a coding assistant for the code, configs and documentation. All design decisions were reviewed by the student and are logged in `DECISIONS.md`.

## Compute and energy

- NTNU IDUN cluster documentation: https://www.hpc.ntnu.no/idun/
- L. Lannelongue, J. Grealey, M. Inouye, *Green Algorithms: Quantifying the Carbon Footprint of Computation*, Advanced Science 8(12) (2021) 2100707. https://doi.org/10.1002/advs.202100707. Used for the energy model, the memory power (0.3725 W/GB), and the global average PUE of 1.67 (which it takes from the Uptime Institute, 2019).
- EV Database, *Energy consumption of full electric vehicles*: average real-world consumption 191 Wh/km. https://ev-database.org/cheatsheet/energy-consumption-electric-car (accessed 2026-11-13).
- CPU power per core from vendor TDPs: Intel Xeon Gold 6148 (150 W, 20 cores), https://www.intel.com/content/www/us/en/products/sku/120489/intel-xeon-gold-6148-processor-27-5m-cache-2-40-ghz/specifications.html; AMD EPYC 7742 (225 W default TDP, 64 cores), https://www.amd.com/en/support/downloads/drivers.html/processors/epyc/epyc-7002-series/amd-epyc-7742.html.
