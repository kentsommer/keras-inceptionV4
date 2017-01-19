
from keras.layers import Input, merge, Dropout, Dense, Flatten, Activation
from keras.layers.convolutional import MaxPooling2D, Convolution2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model

from keras import backend as K

#########################################################################################
# Implements the Inception Network v4 (http://arxiv.org/pdf/1602.07261v1.pdf) in Keras. #
#########################################################################################


def block_inception_a(input):
    if K.image_dim_ordering() == "th":
        channel_axis = 1
    else:
        channel_axis = -1

    branch_0 = Convolution2D(96, 1, 1, activation='relu', border_mode='same')(input)

    branch_1 = Convolution2D(64, 1, 1, activation='relu', border_mode='same')(input)
    branch_1 = Convolution2D(96, 3, 3, activation='relu', border_mode='same')(branch_1)

    branch_2 = Convolution2D(64, 1, 1, activation='relu', border_mode='same')(input)
    branch_2 = Convolution2D(96, 3, 3, activation='relu', border_mode='same')(branch_2)
    branch_2 = Convolution2D(96, 3, 3, activation='relu', border_mode='same')(branch_2)

    branch_3 = AveragePooling2D((3,3), strides=(1,1), border_mode='same')(input)
    branch_3 = Convolution2D(96, 1, 1, activation='relu', border_mode='same')(branch_3)

    x = merge([branch_0, branch_1, branch_2, branch_3], mode='concat', concat_axis=channel_axis)
    # x = BatchNormalization(axis=channel_axis)(x)
    # x = Activation('relu')(x)
    return x


def block_reduction_a(input):
    if K.image_dim_ordering() == "th":
        channel_axis = 1
    else:
        channel_axis = -1

    branch_0 = Convolution2D(384, 3, 3, activation='relu', subsample=(2,2), border_mode='valid')(input)

    branch_1 = Convolution2D(192, 1, 1, activation='relu', border_mode='same')(input)
    branch_1 = Convolution2D(224, 3, 3, activation='relu', border_mode='same')(branch_1)
    branch_1 = Convolution2D(256, 3, 3, activation='relu', subsample=(2,2), border_mode='valid')(branch_1)

    branch_2 = MaxPooling2D((3,3), strides=(2,2), border_mode='valid')(input)

    x = merge([branch_0, branch_1, branch_2], mode='concat', concat_axis=channel_axis)
    # x = BatchNormalization(axis=channel_axis)(x)
    # x = Activation('relu')(x)
    return x


def block_inception_b(input):
    if K.image_dim_ordering() == "th":
        channel_axis = 1
    else:
        channel_axis = -1

    branch_0 = Convolution2D(384, 1, 1, activation='relu', border_mode='same')(input)

    branch_1 = Convolution2D(192, 1, 1, activation='relu', border_mode='same')(input)
    branch_1 = Convolution2D(224, 1, 7, activation='relu', border_mode='same')(branch_1)
    branch_1 = Convolution2D(256, 7, 1, activation='relu', border_mode='same')(branch_1)

    branch_2 = Convolution2D(192, 1, 1, activation='relu', border_mode='same')(input)
    branch_2 = Convolution2D(192, 7, 1, activation='relu', border_mode='same')(branch_2)
    branch_2 = Convolution2D(224, 1, 7, activation='relu', border_mode='same')(branch_2)
    branch_2 = Convolution2D(224, 7, 1, activation='relu', border_mode='same')(branch_2)
    branch_2 = Convolution2D(256, 1, 7, activation='relu', border_mode='same')(branch_2)

    branch_3 = AveragePooling2D((3,3), strides=(1,1), border_mode='same')(input)
    branch_3 = Convolution2D(128, 1, 1, activation='relu', border_mode='same')(branch_3)

    x = merge([branch_0, branch_1, branch_2, branch_3], mode='concat', concat_axis=channel_axis)
    # x = BatchNormalization(axis=channel_axis)(x)
    # x = Activation('relu')(x)
    return x


def block_reduction_b(input):
    if K.image_dim_ordering() == "th":
        channel_axis = 1
    else:
        channel_axis = -1

    branch_0 = Convolution2D(192, 1, 1, activation='relu')(input)
    branch_0 = Convolution2D(192, 3, 3, activation='relu', subsample=(2, 2), border_mode='valid')(branch_0)

    branch_1 = Convolution2D(256, 1, 1, activation='relu', border_mode='same')(input)
    branch_1 = Convolution2D(256, 1, 7, activation='relu', border_mode='same')(branch_1)
    branch_1 = Convolution2D(320, 7, 1, activation='relu', border_mode='same')(branch_1)
    branch_1 = Convolution2D(320, 3, 3, activation='relu', subsample=(2,2), border_mode='valid')(branch_1)

    branch_2 = MaxPooling2D((3, 3), strides=(2, 2), border_mode='valid')(input)

    x = merge([branch_0, branch_1, branch_2], mode='concat', concat_axis=channel_axis)
    # x = BatchNormalization(axis=channel_axis)(x)
    # x = Activation('relu')(x)
    return x


def block_inception_c(input):
    if K.image_dim_ordering() == "th":
        channel_axis = 1
    else:
        channel_axis = -1

    branch_0 = Convolution2D(256, 1, 1, activation='relu', border_mode='same')(input)


    branch_1 = Convolution2D(384, 1, 1, activation='relu', border_mode='same')(input)
    branch_1 = merge([Convolution2D(256, 1, 3, activation='relu', border_mode='same')(branch_1),
					  Convolution2D(256, 3, 1, activation='relu', border_mode='same')(branch_1)], 
				  mode='concat', concat_axis=channel_axis)
    # c3_1 = Convolution2D(256, 1, 3, activation='relu', border_mode='same')(c3)
    # c3_2 = Convolution2D(256, 3, 1, activation='relu', border_mode='same')(c3)


    branch_2 = Convolution2D(384, 1, 1, activation='relu', border_mode='same')(input)
    branch_2 = Convolution2D(448, 3, 1, activation='relu', border_mode='same')(branch_2)
    branch_2 = Convolution2D(512, 1, 3, activation='relu', border_mode='same')(branch_2)
    branch_2 = merge([Convolution2D(256, 1, 3, activation='relu', border_mode='same')(branch_2),
					  Convolution2D(256, 3, 1, activation='relu', border_mode='same')(branch_2)], 
				  mode='concat', concat_axis=channel_axis)
    # branch_2 = Convolution2D(256, 1, 3, activation='relu', border_mode='same')(branch_2)
    # branch_2 = Convolution2D(256, 3, 1, activation='relu', border_mode='same')(branch_2)

    branch_3 = AveragePooling2D((3, 3), strides=(1, 1), border_mode='same')(input)
    branch_3 = Convolution2D(256, 1, 1, activation='relu', border_mode='same')(branch_3)

    x = merge([branch_0, branch_1, branch_2, branch_3], mode='concat', concat_axis=channel_axis)
    # x = BatchNormalization(axis=channel_axis)(x)
    # x = Activation('relu')(x)
    return x


def inception_v4_base(input):
    if K.image_dim_ordering() == "th":
        channel_axis = 1
    else:
        channel_axis = -1

    # Input Shape is 299 x 299 x 3 (th) or 3 x 299 x 299 (th)
    net = Convolution2D(32, 3, 3, activation='relu', subsample=(2,2), border_mode='valid')(input)
    net = Convolution2D(32, 3, 3, activation='relu', border_mode='valid')(net)
    net = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(net)

    branch_0 = MaxPooling2D((3,3), strides=(2,2), border_mode='valid')(net)

    branch_1 = Convolution2D(96, 3, 3, activation='relu', subsample=(2,2), border_mode='valid')(net)

    net = merge([branch_0, branch_1], mode='concat', concat_axis=channel_axis)

    branch_0 = Convolution2D(64, 1, 1, activation='relu', border_mode='same')(net)
    branch_0 = Convolution2D(96, 3, 3, activation='relu', border_mode='valid')(branch_0)

    branch_1 = Convolution2D(64, 1, 1, activation='relu', border_mode='same')(net)
    branch_1 = Convolution2D(64, 1, 7, activation='relu', border_mode='same')(branch_1)
    branch_1 = Convolution2D(64, 7, 1, activation='relu', border_mode='same')(branch_1)
    branch_1 = Convolution2D(96, 3, 3, activation='relu', border_mode='valid')(branch_1)

    net = merge([branch_0, branch_1], mode='concat', concat_axis=channel_axis)

    branch_0 = Convolution2D(192, 3, 3, activation='relu', subsample=(2,2), border_mode='valid')(net)
    branch_1 = MaxPooling2D((3,3), strides=(2,2), border_mode='valid')(net)

    net = merge([branch_0, branch_1], mode='concat', concat_axis=channel_axis)
    # net = BatchNormalization(axis=channel_axis)(net)
    # net = Activation('relu')(net)

    # 35 x 35 x 384
    # 4 x Inception-A blocks
    for idx in xrange(4):
    	net = block_inception_a(net)

    # 35 x 35 x 384
    # Reduction-A block
    net = block_reduction_a(net)

    # 17 x 17 x 1024
    # 7 x Inception-B blocks
    for idx in xrange(7):
    	net = block_inception_b(net)

    # 17 x 17 x 1024
    # Reduction-B block
    net = block_reduction_b(net)

    # 8 x 8 x 1536
    # 3 x Inception-C blocks
    for idx in xrange(3):
    	net = block_inception_c(net)

    return net


def inception_v4(num_classes, dropout_keep_prob):
    '''
    Creates the inception v4 network

    Args:
    	num_classes: number of classes
    	dropout_keep_prob: float, the fraction to keep before final layer.
    
    Returns: 
    	logits: the logits outputs of the model.
    '''

    # Input Shape is 299 x 299 x 3 (tf) or 3 x 299 x 299 (th)
    if K.image_dim_ordering() == 'th':
        inputs = Input((3, 299, 299))
    else:
        inputs = Input((299, 299, 3))

    # Make inception base
    net = inception_v4_base(inputs)


    # Final pooling and prediction

    # 8 x 8 x 1536
    net = AveragePooling2D((8,8), border_mode='valid')(net)

    # 1 x 1 x 1536
    net = Dropout(dropout_keep_prob)(net)
    net = Flatten()(net)

    # 1536
    predictions = Dense(output_dim=num_classes, activation='softmax')(net)

    model = Model(inputs, predictions, name='inception_v4')
    return model


def create_model(num_classes=1000, dropout_keep_prob=0.8):
	return inception_v4(num_classes, dropout_keep_prob)