# A 3D Point-Cloud Dataset of Prehistoric Pottery Impressions

Official repository for the paper:

> **A 3D Point-Cloud Dataset of Prehistoric Pottery Impressions, with Size-Aware Classification and Denoising Baselines**
> Rio Fukuda, Masayoshi Aritsugi, Hiroki Obata, and Israel Mendonça
> *ACM Journal on Computing and Cultural Heritage (JOCCH)*, 2026. (https://doi.org/10.1145/3859284)

[![Data: CC BY 4.0](https://img.shields.io/badge/Data-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Code: MIT](https://img.shields.io/badge/Code-MIT-blue.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23788644.svg)](https://doi.org/10.5281/zenodo.23788644)
-->

## Overview

Pottery excavated from Japan's Jōmon-Yayoi period (ca. 16,500–2,800 BP) preserves impressions of seeds and insects that offer rare evidence of early agriculture and food storage. This repository hosts the **code** for the baselines in our paper: size-aware classification (SA-PointNet++) and morphological denoising (Uniform Erosion, Adaptive Neck Removal, and Noise Verification) on a 3D point-cloud dataset of such impressions, acquired non-destructively by X-ray CT.

The **dataset itself is archived on Zenodo** (see [Dataset](#dataset)).

## Dataset

The dataset (STL meshes, the fixed data splits, and `metadata.csv`) is archived on Zenodo:

**DOI: [10.5281/zenodo.23788644](https://zenodo.org/records/23788644)** 

The archive has its own README describing the file structure, naming, and metadata; see also Section 3 of the paper.

| Class | Experimental | Jōmon-Yayoi | Noise-Pair |
|---|---:|---:|---:|
| Foxtail Millet | 151 | 81 | — |
| Rice | 155 | 44 | — |
| Perilla | 152 | 25 | — |
| Ailanthoides | 154 | 24 | — |
| Maize Weevil | 151 | 27 | 20 pairs |
| **Total** | **763** | **201** | **20 pairs** |

- **Experimental**: laboratory replicas, split into 663 train / 50 validation / 50 test.
- **Jōmon-Yayoi**: genuine archaeological specimens, used **only for testing** (cross-domain setting).
- **Noise-Pair**: 20 noisy Maize Weevil impressions and their expert-cleaned counterparts, used for denoising parameter optimization and for training the noise verification model.

After downloading the archive from Zenodo, extract it so that the layout is:

```
impression3d/
|-- metadata.csv
|-- splits/              # train.txt, val.txt, test.txt, jomon_test.txt
|-- experimental/        # five class folders
|-- jomon_yayoi/         # five class folders
`-- noise_pair/          # noisy/ and clean/
```

## Code

The code accompanying our paper, "A 3D Point-Cloud Dataset of Prehistoric Pottery Impressions, with Size-Aware Classification and Denoising Baselines" (ACM Journal on Computing and Cultural Heritage), is currently being prepared for release. We will update this repository once it is available. If you have any questions in the meantime, please contact fukuda@st.cs.kumamoto-u.ac.jp.

The release will cover the baselines reported in the paper and the denoising pipeline (Uniform Erosion, Adaptive Neck Removal, and Noise Verification).

### Environment

The classification baselines are built on PointNet++ and PointNeXt. Please set up the environments for these two models by following the instructions in the original authors' code:

- PointNet++(PyTorch implementation): <https://github.com/yanx27/Pointnet_Pointnet2_pytorch>
- PointNeXt(official implementation): <https://github.com/guochengqian/PointNeXt>

## License

- **Code** (this repository): [MIT License](LICENSE)
- **Dataset** (Zenodo): [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)

## Citation

If you use this dataset or code, please cite:

<!-- TODO: add volume, number, articleno and doi once the JOCCH record is available -->

```bibtex
@article{fukuda2026impressions,
  author  = {Fukuda, Rio and Aritsugi, Masayoshi and Obata, Hiroki and Mendon{\c{c}}a, Israel},
  title   = {A 3D Point-Cloud Dataset of Prehistoric Pottery Impressions, with Size-Aware Classification and Denoising Baselines},
  journal = {ACM Journal on Computing and Cultural Heritage},
  year    = {2026},
  doi     = {10.1145/3859284}
}
```

## Acknowledgments

This work was part of the DRA project funded by the Japan Society for the Promotion of Science (JSPS) KAKENHI Grant-in-Aid for Scientific Research (A), Grant No. JP24H00107. We thank Ms. Kaina Kobayashi and Ms. Naoko Hatakeyama (Obata Laboratory, Kumamoto University) for X-ray CT scanning, expert identification and classification of the impressions, and manual cleaning of the noisy specimens.
