import tensorflow as tf
from model.residual_block import make_basic_block_layer, make_bottleneck_layer
from model.cnn import TransformerLayer
from tensorflow.keras.layers import Conv1D, Activation, Dense, concatenate, BatchNormalization, GlobalAveragePooling1D, Input, MaxPooling1D, Lambda, GlobalAveragePooling2D, ReLU, MaxPooling2D, Flatten, Dropout, LSTM


def lstm_model(x):
  x = LSTM(units=64, return_sequences=True)(x)
  x = tf.keras.activations.tanh(x)
  x = Dropout(0.2)(x)
  x = GlobalAveragePooling1D(data_format='channels_first', keepdims=False)(x)
  x = tf.keras.activations.tanh(x)
  x = Dropout(0.2)(x)
  return x

class ResNetTypeI(tf.keras.Model):
    def __init__(self, opt, layer_params):
        super(ResNetTypeI, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(filters=64,
                                            kernel_size=(7, 7),
                                            strides=2,
                                            padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                               strides=2,
                                               padding="same")

        self.layer1 = make_basic_block_layer(filter_num=64,
                                             blocks=layer_params[0])
        self.layer2 = make_basic_block_layer(filter_num=128,
                                             blocks=layer_params[1],
                                             stride=2)
        self.layer3 = make_basic_block_layer(filter_num=256,
                                             blocks=layer_params[2],
                                             stride=2)
        self.layer4 = make_basic_block_layer(filter_num=512,
                                             blocks=layer_params[3],
                                             stride=2)

        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc = tf.keras.layers.Dense(units=opt.num_classes, activation=tf.keras.activations.softmax)

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool1(x)
        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.layer4(x, training=training)
        x = self.avgpool(x)
        output = self.fc(x)

        return output


class ResNetTypeII(tf.keras.Model):
    def __init__(self, opt, layer_params):
        super(ResNetTypeII, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=64,
                                            kernel_size=(7, 7),
                                            strides=2,
                                            padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                               strides=2,
                                               padding="same")

        self.layer1 = make_bottleneck_layer(filter_num=64,
                                            blocks=layer_params[0])
        self.layer2 = make_bottleneck_layer(filter_num=128,
                                            blocks=layer_params[1],
                                            stride=2)
        self.layer3 = make_bottleneck_layer(filter_num=256,
                                            blocks=layer_params[2],
                                            stride=2)
        self.layer4 = make_bottleneck_layer(filter_num=512,
                                            blocks=layer_params[3],
                                            stride=2)

        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()
        # self.TransformerLayer = TransformerLayer
        self.lstm_model = lstm_model
        self.expand_dims = tf.expand_dims
        self.fc = tf.keras.layers.Dense(units=opt.num_classes, activation=tf.keras.activations.sigmoid)

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool1(x)
        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.layer4(x, training=training)
        x = self.avgpool(x)
        # x = self.expand_dims(x, -1)
        # x = self.lstm_model(x)
        # x = self.TransformerLayer(x)
        # output = self.fc(x)
        output = x
        return output


def resnet_18(opt):
    return ResNetTypeI(opt, layer_params=[2, 2, 2, 2])


def resnet_34(opt):
    return ResNetTypeI(opt, layer_params=[3, 4, 6, 3])


def resnet_50(opt):
    return ResNetTypeII(opt, layer_params=[3, 4, 6, 3])


def resnet_101(opt):
    return ResNetTypeII(opt, layer_params=[3, 4, 23, 3])


def resnet_152(opt):
    return ResNetTypeII(opt, layer_params=[3, 8, 36, 3])
