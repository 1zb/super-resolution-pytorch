# (WIP) Deep Learning-based Single Image Super Resolution
Implemented with [PyTorch](https://github.com/pytorch/pytorch)

## Introduction
    import net
    srcnn_ex = net.srcnn(f2=5)
    print(srcnn_ex)
    import utils
    print('total parameters: {}'.format(utils.count_parameters(srcnn_ex)))

The output should given:

    Sequential (
      (0): Conv2d(1, 64, kernel_size=(9, 9), stride=(1, 1), padding=(4, 4), bias=False)
      (1): ReLU (inplace)
      (2): Conv2d(64, 32, kernel_size=(5, 5), stride=(1, 1), bias=False)
      (3): ReLU (inplace)
      (4): Conv2d(32, 1, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)
    )
    total parameters: 57184


## TODO
* Add training and testing scripts
* Add more models
* Add function for visualizing the networks

### Models

- [x] SRCNN
- [x] FSRCNN
- [ ] DRCN
- [ ] VDSR
- [ ] SubPixel
- [x] SRGAN (SRResNet)
- [ ] EnhanceNet
- [x] LapSRN

## References
[1] Chao Dong, Chen Change Loy, Kaiming He, Xiaoou Tang, Learning a deep convolutional network for image super-resolution, ECCV, 2014.

[2] Chao Dong, Chen Change Loy, Xiaoou Tang, Accelerating the Super-Resolution Convolutional Neural Network, ECCV, 2016.

[3] Christian Ledig, Lucas Theis, Ferenc Huszar, Jose Caballero, Andrew Cunningham, Alejandro Acosta, Andrew Aitken, Alykhan Tejani, Johannes Totz, Zehan Wang, Wenzhe Shi, Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network, arXiv, 2016.

[4] Wei-Sheng Lai, Jia-Bin Huang, Narendra Ahuja, and Ming-Hsuan Yang, Deep Laplacian Pyramid Networks for Fast and Accurate Super-Resolution, CVPR, 2017.
