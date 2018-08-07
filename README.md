# MobileNet_Zoo
A Keras implementation of [MobileNet_V1](https://arxiv.org/abs/1704.04861) and [MobileNet_V2](https://arxiv.org/abs/1801.04381).

## Requirement
- OpenCV 3
- Python 3
- Tensorflow-gpu 
- Keras

## MobileNet v2

**MobileNet v2:**  

Each line describes a sequence of 1 or more identical (modulo stride) layers, repeated n times. All layers in the same sequence have the same number c of output channels. The first layer of each sequence has a stride s and all others use stride 1. All spatial convolutions use 3 X 3 kernels. The expansion factor t is always applied to the input size.

## Reference

	@article{MobileNet_V1,  
	  title={MobileNets:Efficient Convolutional Neural Networks for Mobile Vision Applications},  
	  author={Google Inc.},
	  journal={arXiv:1704.04861},
	  year={2017}
	}
	@article{MobileNet_V2,  
	  title={MobileNetV2: Inverted Residuals and Linear Bottlenecks},  
	  author={Google Inc.},
	  journal={arXiv:1801.04381},
	  year={2018}
	}
