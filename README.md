# MobileNet_Zoo
A Keras implementation of MobileNet_V1 and MobileNet_V2.

## Requirement
- OpenCV 3
- Python 3
- Tensorflow-gpu 
- Keras


## MobileNet v2 and inverted residual block architectures

**MobileNet v2:**  

Each line describes a sequence of 1 or more identical (modulo stride) layers, repeated n times. All layers in the same sequence have the same number c of output channels. The first layer of each sequence has a stride s and all others use stride 1. All spatial convolutions use 3 X 3 kernels. The expansion factor t is always applied to the input size.

## Reference

	@article{MobileNetv2,  
	  title={Inverted Residuals and Linear Bottlenecks Mobile Networks for Classification, Detection and Segmentatio},  
	  author={Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen},
	  journal={arXiv preprint arXiv:1801.04381},
	  year={2018}
	}
