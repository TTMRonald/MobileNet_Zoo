"""
MobileNet_V1
The following table describes the size and accuracy of the 100% MobileNet_V1
on size 224 x 224:
----------------------------------------------------------------------------
Width Multiplier (alpha) | ImageNet Acc |  Multiply-Adds (M) |  Params (M)
----------------------------------------------------------------------------
|   1.0 MobileNet_V1-224    |    70.6 %     |        529        |     4.2     |
|   0.75 MobileNet_V1-224   |    68.4 %     |        325        |     2.6     |
|   0.50 MobileNet_V1-224   |    63.7 %     |        149        |     1.3     |
|   0.25 MobileNet_V1-224   |    50.6 %     |        41         |     0.5     |
----------------------------------------------------------------------------

The following table describes the performance of
the 100 % MobileNet_V1 on various input sizes:
------------------------------------------------------------------------
      Resolution      | ImageNet Acc | Multiply-Adds (M) | Params (M)
------------------------------------------------------------------------
|  1.0 MobileNet_V1-224  |    70.6 %    |        529        |     4.2     |
|  1.0 MobileNet_V1-192  |    69.1 %    |        529        |     4.2     |
|  1.0 MobileNet_V1-160  |    67.2 %    |        529        |     4.2     |
|  1.0 MobileNet_V1-128  |    64.4 %    |        529        |     4.2     |
------------------------------------------------------------------------
"""

from keras.applications.imagenet_utils import _obtain_input_shape
from keras.models import Model
from keras import backend as K, initializers, regularizers, constraints
from keras.layers import Input, Convolution2D, GlobalAveragePooling2D, Dropout, BatchNormalization, Activation, add, Reshape, Dense
from keras.engine.topology import get_source_inputs
from keras.backend import image_data_format
from keras.backend.tensorflow_backend import _preprocess_conv2d_input, _preprocess_padding
from keras.engine.topology import InputSpec
from keras.utils import conv_utils
from keras.utils.vis_utils import plot_model
import tensorflow as tf

class DepthwiseConvolution2D(Convolution2D):

    def __init__(self,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 depth_multiplier=1,
                 activation=None,
                 use_bias=True,
                 depthwise_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 depthwise_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 depthwise_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(DepthwiseConvolution2D, self).__init__(
            filters=None,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            activation=activation,
            use_bias=use_bias,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            bias_constraint=bias_constraint,
            **kwargs)

        self.depth_multiplier = depth_multiplier
        self.depthwise_initializer = initializers.get(depthwise_initializer)
        self.depthwise_regularizer = regularizers.get(depthwise_regularizer)
        self.depthwise_constraint = constraints.get(depthwise_constraint)
        self.bias_initializer = initializers.get(bias_initializer)

    def build(self, input_shape):
        if len(input_shape) < 4:
            raise ValueError('Inputs to `SeparableConv2D` should have rank 4. '
                             'Received input shape:', str(input_shape))
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = 3
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs to '
                             '`SeparableConv2D` '
                             'should be defined. Found `None`.')
        input_dim = int(input_shape[channel_axis])
        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)

        self.depthwise_kernel = self.add_weight(
            shape=depthwise_kernel_shape,
            initializer=self.depthwise_initializer,
            name='depthwise_kernel',
            regularizer=self.depthwise_regularizer,
            constraint=self.depthwise_constraint)

        if self.use_bias:
            self.bias = self.add_weight(shape=(input_dim * self.depth_multiplier,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.input_spec = InputSpec(ndim=4, axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs, training=None):
        outputs = K.depthwise_conv2d(
            inputs,
            self.depthwise_kernel,
            strides=self.strides,
            padding=self.padding,
            dilation_rate=self.dilation_rate,
            data_format=self.data_format)

        if self.bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            rows = input_shape[2]
            cols = input_shape[3]
            out_filters = input_shape[1] * self.depth_multiplier
        elif self.data_format == 'channels_last':
            rows = input_shape[1]
            cols = input_shape[2]
            out_filters = input_shape[3] * self.depth_multiplier

        rows = conv_utils.conv_output_length(rows, self.kernel_size[0],
                                             self.padding,
                                             self.strides[0])
        cols = conv_utils.conv_output_length(cols, self.kernel_size[1],
                                             self.padding,
                                             self.strides[1])
        if self.data_format == 'channels_first':
            return (input_shape[0], out_filters, rows, cols)
        elif self.data_format == 'channels_last':
            return (input_shape[0], rows, cols, out_filters)

    def get_config(self):
        config = super(DepthwiseConvolution2D, self).get_config()
        config.pop('filters')
        config.pop('kernel_initializer')
        config.pop('kernel_regularizer')
        config.pop('kernel_constraint')
        config['depth_multiplier'] = self.depth_multiplier
        config['depthwise_initializer'] = initializers.serialize(self.depthwise_initializer)
        config['depthwise_regularizer'] = regularizers.serialize(self.depthwise_regularizer)
        config['depthwise_constraint'] = constraints.serialize(self.depthwise_constraint)
        return config

def relu6(x):
    return K.relu(x, max_value=6)

def conv_block(inputs, filters, kernel, strides):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    x = Convolution2D(filters, kernel, padding='same', strides=strides)(inputs)
    x = BatchNormalization(axis=channel_axis)(x)
    return Activation(relu6)(x)


def bottleneck(inputs, filters, kernel, expand, s, r=False):

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    tchannel = K.int_shape(inputs)[channel_axis] * expand

    x = conv_block(inputs, tchannel, (1, 1), (1, 1))

    x = DepthwiseConvolution2D(kernel, strides=(s, s), depth_multiplier=1, padding='same')(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation(relu6)(x)

    x = Convolution2D(filters, (1, 1), strides=(1, 1), padding='same')(x)
    x = BatchNormalization(axis=channel_axis)(x)

    if r:
        x = add([x, inputs])
    return x


def inverted_residual_block(inputs, filters, kernel, expand, strides, block_id):
    x = bottleneck(inputs, filters, kernel, expand, strides)
    for i in range(1, block_id):
        x = bottleneck(x, filters, kernel, expand, 1, True)
    return x


def MobileNet_V2(input_tensor=None, input_shape=None, alpha=1, classes=1000):
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=96,
                                      data_format=K.image_data_format(),
                                      require_flatten=True)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    x = conv_block(img_input, 32, (3, 3), strides=(2, 2))

    x = inverted_residual_block(x, 16, (3, 3), expand=1, strides=1, block_id=1)
    x = inverted_residual_block(x, 24, (3, 3), expand=6, strides=2, block_id=2)
    x = inverted_residual_block(x, 32, (3, 3), expand=6, strides=2, block_id=3)
    x = inverted_residual_block(x, 64, (3, 3), expand=6, strides=2, block_id=4)
    x = inverted_residual_block(x, 96, (3, 3), expand=6, strides=1, block_id=3)
    x = inverted_residual_block(x, 160, (3, 3), expand=6, strides=2, block_id=3)
    x = inverted_residual_block(x, 320, (3, 3), expand=6, strides=1, block_id=1)

    x = conv_block(x, 1280, (1, 1), strides=(1, 1))

    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5, name='dropout')(x)
    outputs = Dense(classes, activation='softmax')(x)

    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    model = Model(inputs, outputs, name='mobilenet_v2')
    # plot_model(model, to_file='images/MobileNet_V2.png', show_shapes=True)

    return model

