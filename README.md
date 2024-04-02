# Multi-view Representation for 3D Shape Completion & Reconstruction

**Render4Completion: Synthesizing Multi-view Depth Maps for 3D Shape Completion.**  
Tao Hu, Zhizhong Han, Abhinav Shrivastava, Matthias Zwicker.
IEEE ICCV Geometry Meets Deep Learning Workshop (ICCVW 2019 [[Paper]](https://openaccess.thecvf.com/content_ICCVW_2019/papers/GMDL/Hu_Render4Completion_Synthesizing_Multi-View_Depth_Maps_for_3D_Shape_Completion_ICCVW_2019_paper.pdf)

* ***Propose multi-view depth maps for shape representation and propose Multi-View Completion Net (MVCN) for 3D shape completion.*** 


**3D Shape Completion with Multi-view Consistent Inference.**  
Tao Hu, Zhizhong Han, Matthias Zwicker.
AAAI Conference on Artificial Intelligence (AAAI 2020) [[Paper]](https://arxiv.org/abs/1911.12465) 

* ***Solve the geometry consistency problem in multi-view representation.***

**Learning to Generate Dense Point Clouds with Textures on Multiple Categories.**  
Tao Hu, Geng Lin, Zhizhong Han, Matthias Zwicker
IEEE Winter Conference on Applications of Computer Vision (WACV 2021) [[Paper]](https://arxiv.org/abs/1912.10545)

* ***Extend the multi-view representation to reconstruct textured point clouds from single RGB images with a two-stage reconstruction pipeline which generalizes well in reconstructing objects from unseen categories.***

This repository contains the code, pre-trained model, and test datasets for the WACV 2021 paper, which is built upon ICCVW 2019 and AAAI 2020 papers. 

## Download pre-trained models and test datasets.

* [Pretrained_models](https://drive.google.com/file/d/1QcLGOCRaRv5sOs21LCNTZuPaIJZvBLhz/view?usp=sharing). Unzip the Pretrained_models.zip to ./checkpoints directory.
* [Test dataset](https://drive.google.com/file/d/1cCEEfVi5_t3Q5erI40fqgkoyXuyK61UH/view?usp=sharing): pix3d_dataset.zip. Unzip it to ./datasets/pix3d/. The dataset contains the input rgb image of seen categories (pix3d_seen) and unseen categories (pix3d_unseen), and ground truth data including sparse (1024 points) and dense (40k points) point clouds.

## Test

* scripts/test_pix3d_depth.sh generates S_d. Single view reconstruction of seen and unseen categories on Pix3D dataset.

* scripts/test_pix3d_texture.sh generates S_{dt}. Single view reconstruction (with texture) of seen and unseen categories on Pix3D dataset.

* scripts/test_pix3d_mix_depth_texture.sh generates S_{d+t} by mixing the depth of S_d and texture of S_{dt} together.

## Samples
We provide sample images and instructions in ./datasets/pix3d/samples.

## Code Reference
1. Parts of the network architecture were built on [Pix2Pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).
2. pc_distance package was borrowed from [PCN](https://github.com/wentaoyuan/pcn)

## License
This project Code is released under the MIT License (refer to the LICENSE file for details).

## Citation
```bibtex
@InProceedings{Hu_2021_WACV,
    author    = {Hu, Tao and Lin, Geng and Han, Zhizhong and Zwicker, Matthias},
    title     = {Learning to Generate Dense Point Clouds With Textures on Multiple Categories},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2021},
    pages     = {2170-2179}
}
```
