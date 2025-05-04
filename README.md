<p align="center" style="font-size: 30px;"><b>Low-Light Image Enhancement via Global-Local Collaborative Transformer</b></p>

> **Abstract:** *Low-light image enhancement aims to refine the lighting conditions of low-light images. However, existing enhancement methods lack an explicit learning mechanism for perceiving exposure, making it difficult to handle unevenly exposed low-light images. To address this issue, we propose a global-local collaborative transformer, an enhancer inspired by image editing experts, which exploits the collaboration between global and local adjustment to adaptively enhance low-light images with complex exposure. Specifically, the proposed model independently learns local and global self-attention to implement the heterogeneous adjustment required for low-light regions with different exposure, and then employs an illumination difference map to dynamically determine the contributions of the two types of adjustment. Meanwhile, we develop a feed-forward neural network integrating globality and locality to further promote the efficacy of the proposed collaborative learning paradigm. In addition, a history-aware contrastive loss is designed to encourage the model to generate high-quality enhanced images that approximate the distribution domain of the normal-light images. Experiments performed on several popular public datasets demonstrate that the proposed enhancer outperforms state-of-the-art models. In particular, it shows competitive performance for low-light images with uneven exposure.* 

# Enhanced Images

![Enhanced Images](./assets/Enhanced-LOLv1.png)

<p align="center">Fig.1. Visual comparison of different enhancers on LOLv1 dataset.</p>

![Enhanced Images](./assets/Enhanced-LOLv2-real.png)

<p align="center">Fig.2. Visual comparison of different enhancers on LOLv2-real dataset.</p>

![Enhanced Images](./assets/Enhanced-LOLv2-synthetic.png)

<p align="center">Fig.3. Visual comparison of different enhancers on LOLv2-synthetic dataset.</p>

# Requirements

*  Python 3.9.17
*  Torch 2.0.0
*  Torchvision 0.15.0
*  Numpy 1.25.2
*  Pillow 9.4.0

# Datasets
*  Training dataset: [LOLv1](https://drive.google.com/file/d/18bs_mAREhLipaM2qvhxs7u7ff2VSHet2/view), [LOLv2-real, LOLv2-synthetic](https://drive.google.com/file/d/1dzuLCk9_gE2bFF222n3-7GVUlSVHpMYC/view)
*  Testing dataset: [MEF, LIME, NPE, DICM](https://drive.google.com/file/d/1YHQ1eW_2WAFSH9vcSY0b933Nzkjpa_tC/view?usp=drive_link)
*  Low-light object detection: [ExDark](https://github.com/cs-chan/Exclusively-Dark-Image-Dataset)



**Acknowledgment:** This code is based on the [Restormer](https://github.com/swz30/Restormer) and the [SNR-Aware](https://github.com/dvlab-research/SNR-Aware-Low-Light-Enhance).