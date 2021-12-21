# 3D-Reconstruction
Pretrained model and test for "Learning to Generate Dense Point Clouds with Textures on Multiple Categories" WACV 2021.
(Under Construction)

## Download pretrained models and test datasets.

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