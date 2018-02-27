from net.utility.file import *
from net.blocks import *
from net.rpn_nms_op import tf_rpn_nms
from net.lib.roi_pooling_layer.roi_pooling_op import roi_pool as tf_roipooling
# from net.roipooling_op import roi_pool as tf_roipooling
from config import cfg
from net.resnet import ResnetBuilder
from keras.models import Model
import keras.applications.xception as xcep
import keras.applications.vgg16 as vgg16
from keras.preprocessing import image
from keras.models import Model
from keras import layers
from keras.layers import (
    Input,
    Activation,
    Dense,
    Flatten,
    SeparableConv2D,
    Conv2D,
    BatchNormalization,
    MaxPooling2D,
    Dropout,
    Average,
    Concatenate
    )
from net.configuration import *
from keras.utils import get_file
from keras import regularizers

# to use focal loss
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops

top_view_rpn_name = 'top_view_rpn'
imfeature_net_name = 'image_feature'
fusion_net_name = 'fusion'


def BatchNormalizationUpdateOps(input, training, name=None):
    cur_bn = BatchNormalization(name=name) if name else BatchNormalization()
    block = cur_bn(input, training=training)
    for update in cur_bn.updates:
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update)
    return block


def top_feature_net(input, anchors, inds_inside, num_bases):
    """temporary net for debugging only. may not follow the paper exactly .... 
    :param input: 
    :param anchors: 
    :param inds_inside: 
    :param num_bases: 
    :return: 
            top_features, top_scores, top_probs, top_deltas, proposals, proposal_scores
    """
    stride=1.
    #with tf.variable_scope('top-preprocess') as scope:
    #    input = input

    with tf.variable_scope('top-block-1') as scope:
        block = conv2d_bn_relu(input, num_kernels=32, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='1')
        block = conv2d_bn_relu(block, num_kernels=32, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='2')
        block = maxpool(block, kernel_size=(2,2), stride=[1,2,2,1], padding='SAME', name='4' )
        stride *=2

    with tf.variable_scope('top-block-2') as scope:
        block = conv2d_bn_relu(block, num_kernels=64, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='1')
        block = conv2d_bn_relu(block, num_kernels=64, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='2')
        block = maxpool(block, kernel_size=(2,2), stride=[1,2,2,1], padding='SAME', name='4' )
        stride *=2

    with tf.variable_scope('top-block-3') as scope:
        block = conv2d_bn_relu(block, num_kernels=128, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='1')
        block = conv2d_bn_relu(block, num_kernels=128, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='2')
        block = conv2d_bn_relu(block, num_kernels=128, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='3')
        block = maxpool(block, kernel_size=(2,2), stride=[1,2,2,1], padding='SAME', name='4' )
        stride *=2

    with tf.variable_scope('top-block-4') as scope:
        block = conv2d_bn_relu(block, num_kernels=128, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='1')
        block = conv2d_bn_relu(block, num_kernels=128, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='2')
        block = conv2d_bn_relu(block, num_kernels=128, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='3')


    with tf.variable_scope('top') as scope:
        #up     = upsample2d(block, factor = 2, has_bias=True, trainable=True, name='1')
        #up     = block
        up      = conv2d_bn_relu(block, num_kernels=128, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='2')
        scores  = conv2d(up, num_kernels=2*num_bases, kernel_size=(1,1), stride=[1,1,1,1], padding='SAME', name='score')
        probs   = tf.nn.softmax( tf.reshape(scores,[-1,2]), name='prob')
        deltas  = conv2d(up, num_kernels=4*num_bases, kernel_size=(1,1), stride=[1,1,1,1], padding='SAME', name='delta')

    #<todo> flip to train and test mode nms (e.g. different nms_pre_topn values): use tf.cond
    with tf.variable_scope('top-nms') as scope:    #non-max
        batch_size, img_height, img_width, img_channel = input.get_shape().as_list()
        img_scale = 1
        rois, roi_scores = tf_rpn_nms( probs, deltas, anchors, inds_inside,
                                       stride, img_width, img_height, img_scale,
                                       nms_thresh=0.7, min_size=stride, nms_pre_topn=500, nms_post_topn=100,
                                       name ='nms')

    #<todo> feature = upsample2d(block, factor = 4,  ...)
    feature = block

    print ('top: scale=%f, stride=%d'%(1./stride, stride))
    return feature, scores, probs, deltas, rois, roi_scores, stride



def top_feature_net_vgg16(input, anchors, inds_inside, num_bases):
    """
    :param input:
    :param anchors:
    :param inds_inside:
    :param num_bases:
    :return:
           feature, scores, probs, deltas, train_rois, train_roi_scores, top_anchors_stride,top_feature_stride, \
           infer_rois, infer_roi_scores
    """
    batch_size, img_height, img_width, img_channel = input.get_shape().as_list()
    top_feature_stride = 8

    # preprocess input, in NHWC, in BGR order
    with tf.variable_scope('rpn-feature-extract') as scope:
        print('build vgg16 for rpn')
        input_block = Input(tensor=input)
        block = Conv2D(32, (3, 3), padding='same', name='block1_conv1')(input)
        block = BatchNormalizationUpdateOps(block, IS_TRAIN_PHASE, name='block1_bn1')
        block = Activation('relu', name='block1_relu1')(block)

        block = Conv2D(32, (3, 3), padding='same', name='block1_conv2')(block)
        block = BatchNormalizationUpdateOps(block, IS_TRAIN_PHASE, name='block1_bn2')
        block = Activation('relu', name='block1_relu2')(block)
        block = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(block)

        # Block 2
        block = Conv2D(64, (3, 3), padding='same', name='block2_conv1')(block)
        block = BatchNormalizationUpdateOps(block, IS_TRAIN_PHASE, name='block2_bn1')
        block = Activation('relu', name='block2_relu1')(block)

        block = Conv2D(64, (3, 3), padding='same', name='block2_conv2')(block)
        block = BatchNormalizationUpdateOps(block, IS_TRAIN_PHASE, name='block2_bn2')
        block = Activation('relu', name='block2_relu2')(block)
        block = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(block)

        # Block 3
        block = Conv2D(128, (3, 3), padding='same', name='block3_conv1')(block)
        block = BatchNormalizationUpdateOps(block, IS_TRAIN_PHASE, name='block3_bn1')
        block = Activation('relu', name='block3_relu1')(block)

        block = Conv2D(128, (3, 3), padding='same', name='block3_conv2')(block)
        block = BatchNormalizationUpdateOps(block, IS_TRAIN_PHASE, name='block3_bn2')
        block = Activation('relu', name='block3_relu2')(block)

        block = Conv2D(128, (3, 3), padding='same', name='block3_conv3')(block)
        block = BatchNormalizationUpdateOps(block, IS_TRAIN_PHASE, name='block3_bn3')
        block = Activation('relu', name='block3_relu3')(block)
        block = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(block)

        # Block 4
        block = Conv2D(256, (3, 3), padding='same', name='block4_conv1')(block)
        block = BatchNormalizationUpdateOps(block, IS_TRAIN_PHASE, name='block4_bn1')
        block = Activation('relu', name='block4_relu1')(block)

        block = Conv2D(256, (3, 3), padding='same', name='block4_conv2')(block)
        block = BatchNormalizationUpdateOps(block, IS_TRAIN_PHASE, name='block4_bn2')
        block = Activation('relu', name='block4_relu2')(block)

        block = Conv2D(256, (3, 3), padding='same', name='block4_conv3')(block)
        block = BatchNormalizationUpdateOps(block, IS_TRAIN_PHASE, name='block4_bn3')
        block = Activation('relu', name='block4_relu3')(block)

        model = Model(inputs=input_block, outputs=block, name='reduced-vgg16')

        # load pre-trained weights
        temp_model = vgg16.VGG16(include_top=False, weights='imagenet')
        temp_conv_weights = temp_model.get_layer('block1_conv1').get_weights()[0]
        temp_conv_bias = temp_model.get_layer('block1_conv1').get_weights()[1]
        random_out = range(0, temp_conv_weights.shape[3], 2)
        temp_conv_weights = temp_conv_weights[:, :, :, random_out]
        temp_conv_weights = temp_conv_weights[:, :, (0,1,2,0,1,2,0), :]
        model.get_layer('block1_conv1').set_weights((temp_conv_weights, temp_conv_bias[random_out]))

        conv_layer_names = ['block1_conv2', 'block2_conv1', 'block2_conv2', 'block3_conv1', 'block3_conv2', 'block3_conv3',
                            'block4_conv1', 'block4_conv2', 'block4_conv3']
        for layer_name in conv_layer_names:
            temp_conv_weights = temp_model.get_layer(layer_name).get_weights()[0]
            temp_conv_bias = temp_model.get_layer(layer_name).get_weights()[1]
            # sample half channels
            random_in = range(0, temp_conv_weights.shape[2], 2)
            random_out = range(0, temp_conv_weights.shape[3], 2)
            temp_conv_weights = temp_conv_weights[:, :, :, random_out]
            temp_conv_weights = temp_conv_weights[:, :, random_in, :]
            temp_conv_bias = temp_conv_bias[random_out]
            model.get_layer(layer_name).set_weights([temp_conv_weights, temp_conv_bias])

        # upsampling
        block = upsample2d(block, factor=4, has_bias=True, trainable=True, name='upsampling')
        block = BatchNormalizationUpdateOps(block, training=IS_TRAIN_PHASE, name='upsample-bn')
        block = Activation('relu', name='upsample-relu')(block)
        feature = block
        top_feature_stride /= 4

    with tf.variable_scope('predict') as scope:
        top_anchors_stride = 2

        scores = Conv2D(filters=2*num_bases, kernel_size=(3, 3), padding='same', name='rpn-score')(block)
        probs = Activation('softmax', name='rpn-probs')(tf.reshape(scores, [-1, 2]))
        deltas = Conv2D(filters=4*num_bases, kernel_size=(3, 3), padding='same', name='rpn-delta')(block)


    with tf.variable_scope('rpn-nms') as scope:

        train_rois, train_roi_scores = tf_rpn_nms( probs, deltas, anchors, inds_inside, \
                                                   top_anchors_stride, img_width, img_height, img_scale=1, \
                                                   nms_thresh=0.7, min_size=top_anchors_stride * 2, \
                                                   nms_pre_topn=CFG.TRAIN.RPN_NMS_PRE_TOPN, nms_post_topn=CFG.TRAIN.RPN_NMS_POST_TOPN, \
                                                   name ='train-rpn-nms')

        infer_rois, infer_roi_scores = tf_rpn_nms( probs, deltas, anchors, inds_inside, \
                                                   top_anchors_stride, img_width, img_height, img_scale=1, \
                                                   nms_thresh=0.7, min_size=top_anchors_stride * 2, \
                                                   nms_pre_topn=CFG.TRAIN.RPN_NMS_PRE_TOPN, nms_post_topn=CFG.TEST.RPN_NMS_POST_TOPN, \
                                                   name ='infer-rpn-nms')


    print ('top: anchor stride=%d, feature_stride=%d'%(top_anchors_stride, top_feature_stride))
    return feature, scores, probs, deltas, train_rois, train_roi_scores, top_anchors_stride, top_feature_stride, \
           infer_rois, infer_roi_scores


def rgb_feature_net_vgg16(input):
    print('build vgg16 for rgb')
    stride = 8
    # preprocess input, in NHWC, in BGR order
    mean = tf.constant([103.939, 116.779, 123.68], dtype=tf.float32)
    input -= mean
    with tf.variable_scope('rgb-feature-extract'):
        input_block = Input(tensor=input)
        block = Conv2D(32, (3, 3), padding='same', name='block1_conv1')(input)
        block = BatchNormalizationUpdateOps(block, IS_TRAIN_PHASE, name='block1_bn1')
        block = Activation('relu', name='block1_relu1')(block)

        block = Conv2D(32, (3, 3), padding='same', name='block1_conv2')(block)
        block = BatchNormalizationUpdateOps(block, IS_TRAIN_PHASE, name='block1_bn2')
        block = Activation('relu', name='block1_relu2')(block)
        block = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(block)

        # Block 2
        block = Conv2D(64, (3, 3), padding='same', name='block2_conv1')(block)
        block = BatchNormalizationUpdateOps(block, IS_TRAIN_PHASE, name='block2_bn1')
        block = Activation('relu', name='block2_relu1')(block)

        block = Conv2D(64, (3, 3), padding='same', name='block2_conv2')(block)
        block = BatchNormalizationUpdateOps(block, IS_TRAIN_PHASE, name='block2_bn2')
        block = Activation('relu', name='block2_relu2')(block)
        block = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(block)

        # Block 3
        block = Conv2D(128, (3, 3), padding='same', name='block3_conv1')(block)
        block = BatchNormalizationUpdateOps(block, IS_TRAIN_PHASE, name='block3_bn1')
        block = Activation('relu', name='block3_relu1')(block)

        block = Conv2D(128, (3, 3), padding='same', name='block3_conv2')(block)
        block = BatchNormalizationUpdateOps(block, IS_TRAIN_PHASE, name='block3_bn2')
        block = Activation('relu', name='block3_relu2')(block)

        block = Conv2D(128, (3, 3), padding='same', name='block3_conv3')(block)
        block = BatchNormalizationUpdateOps(block, IS_TRAIN_PHASE, name='block3_bn3')
        block = Activation('relu', name='block3_relu3')(block)
        block = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(block)

        # Block 4
        block = Conv2D(256, (3, 3), padding='same', name='block4_conv1')(block)
        block = BatchNormalizationUpdateOps(block, IS_TRAIN_PHASE, name='block4_bn1')
        block = Activation('relu', name='block4_relu1')(block)

        block = Conv2D(256, (3, 3), padding='same', name='block4_conv2')(block)
        block = BatchNormalizationUpdateOps(block, IS_TRAIN_PHASE, name='block4_bn2')
        block = Activation('relu', name='block4_relu2')(block)

        block = Conv2D(256, (3, 3), padding='same', name='block4_conv3')(block)
        block = BatchNormalizationUpdateOps(block, IS_TRAIN_PHASE, name='block4_bn3')
        block = Activation('relu', name='block4_relu3')(block)

        model = Model(inputs=input_block, outputs=block, name='reduced-vgg16')

        # load pre-trained weights
        temp_model = vgg16.VGG16(include_top=False, weights='imagenet')
        temp_conv_weights = temp_model.get_layer('block1_conv1').get_weights()[0]
        temp_conv_bias = temp_model.get_layer('block1_conv1').get_weights()[1]
        random_out = range(0, temp_conv_weights.shape[3], 2)
        temp_conv_weights = temp_conv_weights[:, :, :, random_out]
        model.get_layer('block1_conv1').set_weights((temp_conv_weights, temp_conv_bias[random_out]))

        conv_layer_names = ['block1_conv2', 'block2_conv1', 'block2_conv2', 'block3_conv1', 'block3_conv2', 'block3_conv3',
                            'block4_conv1', 'block4_conv2', 'block4_conv3']
        for layer_name in conv_layer_names:
            temp_conv_weights = temp_model.get_layer(layer_name).get_weights()[0]
            temp_conv_bias = temp_model.get_layer(layer_name).get_weights()[1]
            # sample half channels
            random_in = range(0, temp_conv_weights.shape[2], 2)
            random_out = range(0, temp_conv_weights.shape[3], 2)
            temp_conv_weights = temp_conv_weights[:, :, :, random_out]
            temp_conv_weights = temp_conv_weights[:, :, random_in, :]
            temp_conv_bias = temp_conv_bias[random_out]
            model.get_layer(layer_name).set_weights([temp_conv_weights, temp_conv_bias])

        # upsampling
        block = upsample2d(block, factor=4, has_bias=True, trainable=True, name='upsampling')
        block = BatchNormalizationUpdateOps(block, training=IS_TRAIN_PHASE, name='upsample-bn')
        block = Activation('relu', name='upsample-relu')(block)
        rgb_featue = block
        stride /= 4
    print ('rgb: stride=%d'%(stride))
    return rgb_featue, stride


def top_feature_net_xcep(input, anchors, inds_inside, num_bases):
    """
    :param input:
    :param anchors:
    :param inds_inside:
    :param num_bases:
    :return:
           feature, scores, probs, deltas, train_rois, train_roi_scores, top_anchors_stride,top_feature_stride, \
           infer_rois, infer_roi_scores
    """
    batch_size, img_height, img_width, img_channel = input.get_shape().as_list()

    with tf.variable_scope('rpn-feature-extract') as scope:
        print('build xcept for rpn')
        block = rpn_xcep_block(input, weights='imagenet')

        block = upsample2d(block, factor=4, has_bias=True, trainable=True, name='upsampling')

        block = BatchNormalizationUpdateOps(block, training=IS_TRAIN_PHASE, name='upsample-bn')
        block = Activation('relu', name='upsample-relu')(block)
        top_feature_stride = 4
        feature = block


    with tf.variable_scope('predict') as scope:
        top_anchors_stride = 4

        block = SeparableConv2D(512, (3, 3), padding='same', use_bias=False, name='final_block_sepconv1_1')(block)
        block = BatchNormalizationUpdateOps(block, training=IS_TRAIN_PHASE, name='final_block_sepconv1_bn_1')
        block = Activation('relu', name='final_block_sepconv1_act_1')(block)

        block = SeparableConv2D(512, (3, 3), padding='same', use_bias=False, name='final_block_sepconv1_2')(block)
        block = BatchNormalizationUpdateOps(block, training=IS_TRAIN_PHASE, name='final_block_sepconv1_bn_2')
        block = Activation('relu', name='final_block_sepconv1_act_2')(block)

        scores = Conv2D(filters=2*num_bases, kernel_size=(3, 3), padding='same', name='rpn-score')(block)
        probs = Activation('softmax', name='rpn-probs')(tf.reshape(scores, [-1, 2]))
        deltas = Conv2D(filters=4*num_bases, kernel_size=(3, 3), padding='same', name='rpn-delta')(block)


    with tf.variable_scope('RPN-NMS') as scope:

        train_rois, train_roi_scores = tf_rpn_nms( probs, deltas, anchors, inds_inside, \
                                                   top_anchors_stride, img_width, img_height, img_scale=1, \
                                                   nms_thresh=0.7, min_size=top_anchors_stride, \
                                                   nms_pre_topn=CFG.TRAIN.RPN_NMS_PRE_TOPN, nms_post_topn=CFG.TRAIN.RPN_NMS_POST_TOPN, \
                                                   name ='train-rpn-nms')

        infer_rois, infer_roi_scores = tf_rpn_nms( probs, deltas, anchors, inds_inside, \
                                                   top_anchors_stride, img_width, img_height, img_scale=1, \
                                                   nms_thresh=0.7, min_size=top_anchors_stride, \
                                                   nms_pre_topn=CFG.TRAIN.RPN_NMS_PRE_TOPN, nms_post_topn=CFG.TEST.RPN_NMS_POST_TOPN, \
                                                   name ='infer-rpn-nms')


    print ('top: anchor stride=%d, feature_stride=%d'%(top_anchors_stride, top_feature_stride))
    return feature, scores, probs, deltas, train_rois, train_roi_scores, top_anchors_stride,top_feature_stride, \
           infer_rois, infer_roi_scores


def top_feature_net_r(input, anchors, inds_inside, num_bases):
    """
    :param input: 
    :param anchors: 
    :param inds_inside: 
    :param num_bases: 
    :return: 
            top_features, top_scores, top_probs, top_deltas, proposals, proposal_scores
    """
    batch_size, img_height, img_width, img_channel = input.get_shape().as_list()

    with tf.variable_scope('feature-extract-resnet') as scope:
        print('build_resnet')
        block = ResnetBuilder.resnet_tiny_smaller_kernel(input)
        feature = block

        top_feature_stride = 4
        # resnet_input = resnet.get_layer('input_1').input
        # resnet_output = resnet.get_layer('add_7').output
        # resnet_f = Model(inputs=resnet_input, outputs=resnet_output)  # add_7
        # # print(resnet_f.summary())
        # block = resnet_f(input)
        block = upsample2d(block, factor=2, has_bias=True, trainable=True, name='upsampling')


    with tf.variable_scope('predict') as scope:
        # block = upsample2d(block, factor=4, has_bias=True, trainable=True, name='1')
        # up     = block
        # kernel_size = config.cfg.TOP_CONV_KERNEL_SIZE
        top_anchors_stride = 2
        block = conv2d_bn_relu(block, num_kernels=128, kernel_size=(1, 1), stride=[1, 1, 1, 1], padding='SAME',
                               name='2')
        scores = conv2d(block, num_kernels=2 * num_bases, kernel_size=(1, 1), stride=[1, 1, 1, 1], padding='SAME',name='score')
        probs = tf.nn.softmax(tf.reshape(scores, [-1, 2]), name='prob')
        deltas = conv2d(block, num_kernels=4 * num_bases, kernel_size=(1, 1), stride=[1, 1, 1, 1], padding='SAME',name='delta')


    with tf.variable_scope('RPN-NMS') as scope:

        train_rois, train_roi_scores = tf_rpn_nms( probs, deltas, anchors, inds_inside, \
                                       top_anchors_stride, img_width, img_height, img_scale=1, \
                                       nms_thresh=0.7, min_size=top_anchors_stride, \
                                       nms_pre_topn=CFG.TRAIN.RPN_NMS_PRE_TOPN, nms_post_topn=CFG.TRAIN.RPN_NMS_POST_TOPN, \
                                       name ='train-rpn-nms')

        infer_rois, infer_roi_scores = tf_rpn_nms( probs, deltas, anchors, inds_inside, \
                                       top_anchors_stride, img_width, img_height, img_scale=1, \
                                       nms_thresh=0.1, min_size=top_anchors_stride, \
                                       nms_pre_topn=CFG.TRAIN.RPN_NMS_PRE_TOPN, nms_post_topn=CFG.TEST.RPN_NMS_POST_TOPN, \
                                       name ='infer-rpn-nms')


    print ('top: anchor stride=%d, feature_stride=%d'%(top_anchors_stride, top_feature_stride))
    return feature, scores, probs, deltas, train_rois, train_roi_scores, top_anchors_stride,top_feature_stride, \
            infer_rois, infer_roi_scores




#------------------------------------------------------------------------------
def rgb_feature_net(input):

    stride=1.
    #with tf.variable_scope('rgb-preprocess') as scope:
    #   input = input-128

    with tf.variable_scope('rgb-block-1') as scope:
        block = conv2d_bn_relu(input, num_kernels=32, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='1')
        block = conv2d_bn_relu(block, num_kernels=32, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='2')
        block = maxpool(block, kernel_size=(2,2), stride=[1,2,2,1], padding='SAME', name='4' )
        stride *=2

    with tf.variable_scope('rgb-block-2') as scope:
        block = conv2d_bn_relu(block, num_kernels=64, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='1')
        block = conv2d_bn_relu(block, num_kernels=64, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='2')
        block = maxpool(block, kernel_size=(2,2), stride=[1,2,2,1], padding='SAME', name='4' )
        stride *=2

    with tf.variable_scope('rgb-block-3') as scope:
        block = conv2d_bn_relu(block, num_kernels=128, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='1')
        block = conv2d_bn_relu(block, num_kernels=128, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='2')
        block = conv2d_bn_relu(block, num_kernels=128, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='3')
        block = maxpool(block, kernel_size=(2,2), stride=[1,2,2,1], padding='SAME', name='4' )
        stride *=2

    with tf.variable_scope('rgb-block-4') as scope:
        block = conv2d_bn_relu(block, num_kernels=128, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='1')
        block = conv2d_bn_relu(block, num_kernels=128, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='2')
        block = conv2d_bn_relu(block, num_kernels=128, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='3')


    #<todo> feature = upsample2d(block, factor = 4,  ...)
    feature = block


    print ('rgb : scale=%f, stride=%d'%(1./stride, stride))
    return feature, stride


def rgb_feature_net_r(input):

    #with tf.variable_scope('rgb-preprocess') as scope:
    #   input = input-128

    batch_size, img_height, img_width, img_channel = input.get_shape().as_list()

    with tf.variable_scope('resnet-block-1') as scope:
        print('build_resnet')
        block = ResnetBuilder.resnet_tiny(input)
        block = conv2d_bn_relu(block, num_kernels=128, kernel_size=(1, 1), stride=[1, 1, 1, 1], padding='SAME', name='2')
        stride = 8

    #<todo> feature = upsample2d(block, factor = 4,  ...)
    feature = block


    print ('rgb : scale=%f, stride=%d'%(1./stride, stride))
    return feature, stride

def rgb_xcep_block(input, weights=None):
    # preprocess input, scale input to [-1, 1]
    input /= 127.5
    input -= 1.

    input_block = Input(tensor=input)
    x = Conv2D(32, (3, 3), strides=(2, 2), use_bias=False, name='block1_conv1')(input_block)
    x = BatchNormalizationUpdateOps(x, training=IS_TRAIN_PHASE, name='block1_conv1_bn')
    x = Activation('relu', name='block1_conv1_act')(x)
    x = Conv2D(64, (3, 3), use_bias=False, name='block1_conv2')(x)
    x = BatchNormalizationUpdateOps(x, training=IS_TRAIN_PHASE, name='block1_conv2_bn')
    x = Activation('relu', name='block1_conv2_act')(x)

    residual = Conv2D(128, (1, 1), strides=(2, 2), \
                      padding='same', use_bias=False)(x)
    residual = BatchNormalizationUpdateOps(residual, training=IS_TRAIN_PHASE)

    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False, name='block2_sepconv1')(x)
    x = BatchNormalizationUpdateOps(x, training=IS_TRAIN_PHASE, name='block2_sepconv1_bn')
    x = Activation('relu', name='block2_sepconv2_act')(x)
    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False, name='block2_sepconv2')(x)
    x = BatchNormalizationUpdateOps(x, training=IS_TRAIN_PHASE, name='block2_sepconv2_bn')

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block2_pool')(x)
    x = layers.add([x, residual])

    residual = Conv2D(256, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalizationUpdateOps(residual, training=IS_TRAIN_PHASE)

    x = Activation('relu', name='block3_sepconv1_act')(x)
    x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='block3_sepconv1')(x)
    x = BatchNormalizationUpdateOps(x, training=IS_TRAIN_PHASE, name='block3_sepconv1_bn')
    x = Activation('relu', name='block3_sepconv2_act')(x)
    x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='block3_sepconv2')(x)
    x = BatchNormalizationUpdateOps(x, training=IS_TRAIN_PHASE, name='block3_sepconv2_bn')

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block3_pool')(x)
    x = layers.add([x, residual])

    residual = Conv2D(728, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalizationUpdateOps(residual, training=IS_TRAIN_PHASE)

    x = Activation('relu', name='block4_sepconv1_act')(x)
    x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='block4_sepconv1')(x)
    x = BatchNormalizationUpdateOps(x, training=IS_TRAIN_PHASE, name='block4_sepconv1_bn')
    x = Activation('relu', name='block4_sepconv2_act')(x)
    x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='block4_sepconv2')(x)
    x = BatchNormalizationUpdateOps(x, training=IS_TRAIN_PHASE, name='block4_sepconv2_bn')

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block4_pool')(x)
    x = layers.add([x, residual])

    for i in range(8):
        residual = x
        prefix = 'block' + str(i + 5)

        x = Activation('relu', name=prefix + '_sepconv1_act')(x)
        x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv1')(x)
        x = BatchNormalizationUpdateOps(x, training=IS_TRAIN_PHASE, name=prefix + '_sepconv1_bn')
        x = Activation('relu', name=prefix + '_sepconv2_act')(x)
        x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv2')(x)
        x = BatchNormalizationUpdateOps(x, training=IS_TRAIN_PHASE, name=prefix + '_sepconv2_bn')
        x = Activation('relu', name=prefix + '_sepconv3_act')(x)
        x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv3')(x)
        x = BatchNormalizationUpdateOps(x, training=IS_TRAIN_PHASE, name=prefix + '_sepconv3_bn')

        x = layers.add([x, residual], name=prefix + '_add')

    feature = x
    stride = 16
    model = Model(input_block, x, name='xception')
    if weights == 'imagenet':
        TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.4/xception_weights_tf_dim_ordering_tf_kernels_notop.h5'
        weights_path = get_file('xception_weights_tf_dim_ordering_tf_kernels_notop.h5', TF_WEIGHTS_PATH_NO_TOP,
                                    cache_subdir='models', file_hash='b0042744bf5b25fce3cb969f33bebb97')
        model.load_weights(weights_path, by_name=True)

    return feature

def rgb_feature_net_xcep(input):
    print('build xcept for rgb')
    stride = 16
    with tf.variable_scope('rgb-feature-extract'):
        block = rgb_xcep_block(input, weights='imagenet')
        block = upsample2d(block, factor=4, has_bias=True, trainable=True, name='upsampling')
        block = BatchNormalizationUpdateOps(block, training=IS_TRAIN_PHASE, name='upsample-bn')
        block = Activation('relu', name='upsample-relu')(block)
        rgb_featue = block
        stride /= 4
    print ('rgb: stride=%d'%(stride))
    return rgb_featue, stride


def rpn_xcep_block(input, weights=None):
    # scale feature to [-1, 1]
    input *= 2.
    input -= 1.

    input_block = Input(tensor=input)
    x = Conv2D(32, (3, 3), strides=(2, 2), use_bias=False, name='block1_conv1_modified')(input_block)
    x = BatchNormalizationUpdateOps(x, training=IS_TRAIN_PHASE, name='block1_conv1_bn')
    x = Activation('relu', name='block1_conv1_act')(x)
    x = Conv2D(64, (3, 3), use_bias=False, name='block1_conv2')(x)
    x = BatchNormalizationUpdateOps(x, training=IS_TRAIN_PHASE, name='block1_conv2_bn')
    x = Activation('relu', name='block1_conv2_act')(x)

    residual = Conv2D(128, (1, 1), strides=(2, 2), \
                      padding='same', use_bias=False)(x)
    residual = BatchNormalizationUpdateOps(residual, training=IS_TRAIN_PHASE)

    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False, name='block2_sepconv1')(x)
    x = BatchNormalizationUpdateOps(x, training=IS_TRAIN_PHASE, name='block2_sepconv1_bn')
    x = Activation('relu', name='block2_sepconv2_act')(x)
    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False, name='block2_sepconv2')(x)
    x = BatchNormalizationUpdateOps(x, training=IS_TRAIN_PHASE, name='block2_sepconv2_bn')

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block2_pool')(x)
    x = layers.add([x, residual])

    residual = Conv2D(256, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalizationUpdateOps(residual, training=IS_TRAIN_PHASE)

    x = Activation('relu', name='block3_sepconv1_act')(x)
    x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='block3_sepconv1')(x)
    x = BatchNormalizationUpdateOps(x, training=IS_TRAIN_PHASE, name='block3_sepconv1_bn')
    x = Activation('relu', name='block3_sepconv2_act')(x)
    x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='block3_sepconv2')(x)
    x = BatchNormalizationUpdateOps(x, training=IS_TRAIN_PHASE, name='block3_sepconv2_bn')

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block3_pool')(x)
    x = layers.add([x, residual])

    residual = Conv2D(728, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalizationUpdateOps(residual, training=IS_TRAIN_PHASE)

    x = Activation('relu', name='block4_sepconv1_act')(x)
    x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='block4_sepconv1')(x)
    x = BatchNormalizationUpdateOps(x, training=IS_TRAIN_PHASE, name='block4_sepconv1_bn')
    x = Activation('relu', name='block4_sepconv2_act')(x)
    x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='block4_sepconv2')(x)
    x = BatchNormalizationUpdateOps(x, training=IS_TRAIN_PHASE, name='block4_sepconv2_bn')

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block4_pool')(x)
    x = layers.add([x, residual])

    for i in range(8):
        residual = x
        prefix = 'block' + str(i + 5)

        x = Activation('relu', name=prefix + '_sepconv1_act')(x)
        x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv1')(x)
        x = BatchNormalizationUpdateOps(x, training=IS_TRAIN_PHASE, name=prefix + '_sepconv1_bn')
        x = Activation('relu', name=prefix + '_sepconv2_act')(x)
        x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv2')(x)
        x = BatchNormalizationUpdateOps(x, training=IS_TRAIN_PHASE, name=prefix + '_sepconv2_bn')
        x = Activation('relu', name=prefix + '_sepconv3_act')(x)
        x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv3')(x)
        x = BatchNormalizationUpdateOps(x, training=IS_TRAIN_PHASE, name=prefix + '_sepconv3_bn')

        x = layers.add([x, residual], name=prefix + '_add')

    feature = x
    stride = 16
    model = Model(input_block, x, name='xception')
    if weights == 'imagenet':
        TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.4/xception_weights_tf_dim_ordering_tf_kernels_notop.h5'
        weights_path = get_file('xception_weights_tf_dim_ordering_tf_kernels_notop.h5', TF_WEIGHTS_PATH_NO_TOP,
                                    cache_subdir='models', file_hash='b0042744bf5b25fce3cb969f33bebb97')
        model.load_weights(weights_path, by_name=True)

    # init the weights of first layer, because the input is not 3 channels (subsample from first conv layer)
    temp_xcep_model = xcep.Xception(include_top=False, weights='imagenet')
    # 3 x 3 x 3 x 32 (HWInOut)
    temp_conv_weights = temp_xcep_model.get_layer('block1_conv1').get_weights()[0]
    indices = (0, 1, 2, 0, 1, 2, 0)
    # 3 x 3 x 7 x 32
    model.get_layer('block1_conv1_modified').set_weights([temp_conv_weights[:,:,indices,:]])

    return feature

#------------------------------------------------------------------------------

# feature_list:
# ( [top_features,     top_rois,     6,6,1./stride],
#   [rgb_features,     rgb_rois,     6,6,1./stride],)
#
def early_fusion(feature_list):
    with tf.variable_scope('fuse-net') as scope:
        num = len(feature_list)
        feature_names = ['top', 'rgb']
        roi_features_list = []
        with tf.variable_scope('roi-pooling'):
            for n in range(num):
                feature = feature_list[n][0]
                roi = feature_list[n][1]
                pool_height = feature_list[n][2]
                pool_width = feature_list[n][3]
                pool_scale = feature_list[n][4]
                if (pool_height == 0 or pool_width == 0): continue

                roi_features, _ = tf_roipooling(feature, roi, pool_height, pool_width,
                                                       pool_scale, name='%s-roi_pooling' % feature_names[n])
                roi_features_list.append(Flatten()(roi_features))

        with tf.variable_scope('early-fusion'):
            # block = Average(name='elementwise-mean')(roi_features_list)
            block = Concatenate(name='concat')(roi_features_list)

            block = Dense(units=2048, name='fc1')(block)
            block = BatchNormalizationUpdateOps(block, training=IS_TRAIN_PHASE, name='bn1')
            block = Activation('relu', name='relu1')(block)
            block = tf.cond(IS_TRAIN_PHASE, lambda: Dropout(rate=0.6, name='dropout1')(block), lambda: block)

            block = Dense(units=2048, name='fc2')(block)
            block = BatchNormalizationUpdateOps(block, training=IS_TRAIN_PHASE, name='bn2')
            block = Activation('relu', name='relu2')(block)
            block = tf.cond(IS_TRAIN_PHASE, lambda: Dropout(rate=0.5, name='dropout2')(block), lambda: block)

            block = Dense(units=2048, name='fc3')(block)
            block = BatchNormalizationUpdateOps(block, training=IS_TRAIN_PHASE, name='bn3')
            block = Activation('relu', name='relu3')(block)
            block = tf.cond(IS_TRAIN_PHASE, lambda: Dropout(rate=0.5, name='dropout3')(block), lambda: block)

    return block



def fusion_net(feature_list, num_class, out_shape=(8,3)):

    with tf.variable_scope('fuse-net') as scope:
        num = len(feature_list)
        feature_names = ['top', 'rgb']
        roi_features_list = []
        for n in range(num):
            feature = feature_list[n][0]
            roi = feature_list[n][1]
            pool_height = feature_list[n][2]
            pool_width = feature_list[n][3]
            pool_scale = feature_list[n][4]
            if (pool_height == 0 or pool_width == 0): continue

            with tf.variable_scope(feature_names[n] + '-roi-pooling'):
                roi_features, roi_idxs = tf_roipooling(feature, roi, pool_height, pool_width,
                                                       pool_scale, name='%s-roi_pooling' % feature_names[n])
            with tf.variable_scope(feature_names[n]+ '-feature-conv'):

                with tf.variable_scope('block1') as scope:
                    block = conv2d_bn_relu(roi_features, num_kernels=128, kernel_size=(3, 3),
                                           stride=[1, 1, 1, 1], padding='SAME',name=feature_names[n]+'_conv_1')
                    residual=block

                    block = conv2d_bn_relu(block, num_kernels=128, kernel_size=(3, 3), stride=[1, 1, 1, 1],
                                           padding='SAME',name=feature_names[n]+'_conv_2')+residual

                    block = avgpool(block, kernel_size=(2, 2), stride=[1, 2, 2, 1],
                                    padding='SAME', name=feature_names[n]+'_avg_pool')
                with tf.variable_scope('block2') as scope:

                    block = conv2d_bn_relu(block, num_kernels=256, kernel_size=(3, 3), stride=[1, 1, 1, 1], padding='SAME',
                                           name=feature_names[n]+'_conv_1')
                    residual = block
                    block = conv2d_bn_relu(block, num_kernels=256, kernel_size=(3, 3), stride=[1, 1, 1, 1], padding='SAME',
                                           name=feature_names[n]+'_conv_2')+residual

                    block = avgpool(block, kernel_size=(2, 2), stride=[1, 2, 2, 1],
                                    padding='SAME', name=feature_names[n]+'_avg_pool')
                with tf.variable_scope('block3') as scope:

                    block = conv2d_bn_relu(block, num_kernels=512, kernel_size=(3, 3), stride=[1, 1, 1, 1], padding='SAME',
                                           name=feature_names[n]+'_conv_1')
                    residual = block
                    block = conv2d_bn_relu(block, num_kernels=512, kernel_size=(3, 3), stride=[1, 1, 1, 1], padding='SAME',
                                           name=feature_names[n]+'_conv_2')+residual

                    block = avgpool(block, kernel_size=(2, 2), stride=[1, 2, 2, 1],
                                    padding='SAME', name=feature_names[n]+'_avg_pool')


                roi_features = flatten(block)
                tf.summary.histogram(feature_names[n], roi_features)
                roi_features_list.append(roi_features)


        with tf.variable_scope('fusion-feature-fc'):
            block = concat(roi_features_list, axis=1, name='concat')

            block = linear_bn_relu(block, num_hiddens=512, name='fc-1')
            block = linear_bn_relu(block, num_hiddens=512, name='fc-2')

    return block


def focal_loss(predictions, labels, gamma=2, alpha=1.0, weights=1.0,
               epsilon=1e-7, scope=None):
  """Adds a Focal Loss term to the training procedure.

  For each value x in `predictions`, and the corresponding l in `labels`,
  the following is calculated:

  ```
    pt = 1 - x                  if l == 0
    pt = x                      if l == 1

    focal_loss = - a * (1 - pt)**g * log(pt)
  ```

  where g is `gamma`, a is `alpha`.

  See: https://arxiv.org/pdf/1708.02002.pdf

  `weights` acts as a coefficient for the loss. If a scalar is provided, then
  the loss is simply scaled by the given value. If `weights` is a tensor of size
  [batch_size], then the total loss for each sample of the batch is rescaled
  by the corresponding element in the `weights` vector. If the shape of
  `weights` matches the shape of `predictions`, then the loss of each
  measurable element of `predictions` is scaled by the corresponding value of
  `weights`.

  Args:
    labels: The ground truth output tensor, same dimensions as 'predictions'.
    predictions: The predicted outputs.
    gamma, alpha: parameters.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
      be either `1`, or the same as the corresponding `losses` dimension).
    epsilon: A small increment to add to avoid taking a log of zero.
    scope: The scope for the operations performed in computing the loss.
    loss_collection: collection to which the loss will be added.
    reduction: Type of reduction to apply to loss.

  Returns:
    Weighted loss float `Tensor`. If `reduction` is `NONE`, this has the same
    shape as `labels`; otherwise, it is scalar.

  Raises:
    ValueError: If the shape of `predictions` doesn't match that of `labels` or
      if the shape of `weights` is invalid.
  """
  with ops.name_scope(scope, "focal_loss",
                      (predictions, labels, weights)) as scope:
    predictions = math_ops.to_float(predictions)
    labels = math_ops.to_float(labels)
    predictions.get_shape().assert_is_compatible_with(labels.get_shape())
    preds = array_ops.where(
        math_ops.equal(labels, 1), predictions, 1. - predictions)


    # debug for pos loss and neg loss
    negs = tf.zeros_like(predictions) + 0.999
    poss = tf.zeros_like(predictions) + 0.999
    pos_preds = array_ops.where(
        math_ops.equal(labels, 1), predictions, negs)
    neg_preds = array_ops.where(
        math_ops.equal(labels, 1), poss, 1 - predictions)
    pos_losses = -alpha * (1. - pos_preds)**gamma * math_ops.log(pos_preds + epsilon)
    neg_losses = -alpha * (1. - neg_preds)**gamma * math_ops.log(neg_preds + epsilon)


    losses = -alpha * (1. - preds)**gamma * math_ops.log(preds + epsilon)
    return tf.losses.compute_weighted_loss(losses, weights, scope=scope), tf.losses.compute_weighted_loss(pos_losses, weights, scope=scope), \
               tf.losses.compute_weighted_loss(neg_losses, weights, scope=scope)

def fuse_loss(scores, deltas, rcnn_labels, rcnn_targets):

    def modified_smooth_l1( deltas, targets, sigma=3.0):
        '''
            ResultLoss = outside_weights * SmoothL1(inside_weights * (box_pred - box_targets))
            SmoothL1(x) = 0.5 * (sigma * x)^2,    if |x| < 1 / sigma^2
                          |x| - 0.5 / sigma^2,    otherwise
        '''
        sigma2 = sigma * sigma
        diffs  =  tf.subtract(deltas, targets)
        smooth_l1_signs = tf.cast(tf.less(tf.abs(diffs), 1.0 / sigma2), tf.float32)

        smooth_l1_option1 = tf.multiply(diffs, diffs) * 0.5 * sigma2
        smooth_l1_option2 = tf.abs(diffs) - 0.5 / sigma2
        smooth_l1_add = tf.multiply(smooth_l1_option1, smooth_l1_signs) + tf.multiply(smooth_l1_option2, 1-smooth_l1_signs)
        smooth_l1 = smooth_l1_add

        return smooth_l1


    _, num_class = scores.get_shape().as_list()
    dim = np.prod(deltas.get_shape().as_list()[1:])//num_class

    with tf.variable_scope('get_scores'):
        rcnn_scores   = tf.nn.softmax(tf.reshape(scores,[-1, num_class], name='rcnn_scores'))
        rcnn_scores = rcnn_scores[:, 1]
        total_loss, pos_loss, neg_loss = focal_loss(predictions=rcnn_scores, labels=rcnn_labels, weights=4.0)
        rcnn_cls_loss = tf.reduce_mean(total_loss)
        pos_rcnn_cls_loss = tf.reduce_mean(pos_loss)
        neg_rcnn_cls_loss = tf.reduce_mean(neg_loss)


        # rcnn_scores   = tf.reshape(scores,[-1, num_class], name='rcnn_scores')
        # rcnn_cls_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        #     logits=rcnn_scores, labels=rcnn_labels))
        # pos_rcnn_cls_loss = tf.constant(0)
        # neg_rcnn_cls_loss = tf.constant(0)



    with tf.variable_scope('get_detals'):
        num = tf.identity( tf.shape(deltas)[0], 'num')
        idx = tf.identity(tf.range(num)*num_class + rcnn_labels,name='idx')
        deltas1      = tf.reshape(deltas,[-1, dim],name='deltas1')

        rcnn_deltas_with_fp  = tf.gather(deltas1,  idx, name='rcnn_deltas_with_fp')  # remove ignore label
        rcnn_targets_with_fp =  tf.reshape(rcnn_targets,[-1, dim], name='rcnn_targets_with_fp')

        #remove false positive
        fp_idxs = tf.where(tf.not_equal(rcnn_labels, 0), name='fp_idxs')
        rcnn_deltas_no_fp  = tf.gather(rcnn_deltas_with_fp,  fp_idxs, name='rcnn_deltas_no_fp')
        rcnn_targets_no_fp =  tf.gather(rcnn_targets_with_fp,  fp_idxs, name='rcnn_targets_no_fp')

    with tf.variable_scope('modified_smooth_l1'):
        rcnn_smooth_l1 = modified_smooth_l1(rcnn_deltas_no_fp, rcnn_targets_no_fp, sigma=3.0)

    rcnn_reg_loss  = tf.reduce_mean(tf.reduce_sum(rcnn_smooth_l1, axis=1))
    return rcnn_cls_loss, rcnn_reg_loss, pos_rcnn_cls_loss, neg_rcnn_cls_loss


def rpn_loss(scores, deltas, inds, pos_inds, rpn_labels, rpn_targets):

    def modified_smooth_l1( box_preds, box_targets, sigma=3.0):
        '''
            ResultLoss = outside_weights * SmoothL1(inside_weights * (box_pred - box_targets))
            SmoothL1(x) = 0.5 * (sigma * x)^2,    if |x| < 1 / sigma^2
                          |x| - 0.5 / sigma^2,    otherwise
        '''
        sigma2 = sigma * sigma
        diffs  =  tf.subtract(box_preds, box_targets)
        smooth_l1_signs = tf.cast(tf.less(tf.abs(diffs), 1.0 / sigma2), tf.float32)

        smooth_l1_option1 = tf.multiply(diffs, diffs) * 0.5 * sigma2
        smooth_l1_option2 = tf.abs(diffs) - 0.  / sigma2
        smooth_l1_add = tf.multiply(smooth_l1_option1, smooth_l1_signs) + tf.multiply(smooth_l1_option2, 1-smooth_l1_signs)
        smooth_l1 = smooth_l1_add   #tf.multiply(box_weights, smooth_l1_add)  #

        return smooth_l1

    scores1      = tf.reshape(scores,[-1,2])
    rpn_scores   = tf.gather(scores1, inds)  # remove ignore label
    # rpn_cls_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_scores, labels=rpn_labels))
    # pos_rpn_cls_loss = tf.constant(0)
    # neg_rpn_cls_loss = tf.constant(0)

    rpn_scores   = tf.nn.softmax(rpn_scores, name='rpn_scores')[:, 1]
    total_loss, pos_loss, neg_loss = focal_loss(predictions=rpn_scores, labels=rpn_labels, weights=4.)
    rpn_cls_loss = tf.reduce_mean(total_loss)
    pos_rpn_cls_loss = tf.reduce_mean(pos_loss)
    neg_rpn_cls_loss = tf.reduce_mean(neg_loss)

    deltas1       = tf.reshape(deltas,[-1,4])
    rpn_deltas    = tf.gather(deltas1, pos_inds)  # remove ignore label

    with tf.variable_scope('modified_smooth_l1'):
        rpn_smooth_l1 = modified_smooth_l1(rpn_deltas, rpn_targets, sigma=3.0)

    rpn_reg_loss  = tf.reduce_mean(tf.reduce_sum(rpn_smooth_l1, axis=1))
    return rpn_cls_loss, rpn_reg_loss, pos_rpn_cls_loss, neg_rpn_cls_loss

def load(top_shape, rgb_shape, num_class, len_bases):

    out_shape = (8, 3)
    stride = 8

    top_anchors = tf.placeholder(shape=[None, 4], dtype=tf.int32, name='anchors')
    top_inside_inds = tf.placeholder(shape=[None], dtype=tf.int32, name='inside_inds')

    top_view = tf.placeholder(shape=[None, *top_shape], dtype=tf.float32, name='top')

    if cfg.RGB_BASENET =='resnet':
        rgb_images = tf.placeholder(shape=[None, *rgb_shape], dtype=tf.float32, name='rgb')
    elif cfg.RGB_BASENET =='xception':
        rgb_images = tf.placeholder(shape=[None, None, None, 3], dtype=tf.float32, name='rgb')
    elif cfg.RGB_BASENET =='vgg16':
        rgb_images = tf.placeholder(shape=[None, None, None, 3], dtype=tf.float32, name='rgb')
    top_rois = tf.placeholder(shape=[None, 5], dtype=tf.float32, name='top_rois')  # todo: change to int32???
    rgb_rois = tf.placeholder(shape=[None, 5], dtype=tf.float32, name='rgb_rois')

    with tf.variable_scope(top_view_rpn_name):
        # top feature
        if cfg.TOP_BASENET == 'resnet':
            top_features, top_scores, top_probs, top_deltas, train_proposals, train_proposal_scores, \
            top_anchors_stride, top_feature_stride, infer_proposals, infer_proposal_scores = \
                top_feature_net_r(top_view, top_anchors, top_inside_inds, len_bases)
        elif cfg.TOP_BASENET == 'xception':
            top_features, top_scores, top_probs, top_deltas, train_proposals, train_proposal_scores, \
            top_anchors_stride, top_feature_stride, infer_proposals, infer_proposal_scores = \
                top_feature_net_xcep(top_view, top_anchors, top_inside_inds, len_bases)
        elif cfg.TOP_BASENET == 'vgg16':
            top_features, top_scores, top_probs, top_deltas, train_proposals, train_proposal_scores, \
            top_anchors_stride, top_feature_stride, infer_proposals, infer_proposal_scores = \
                top_feature_net_vgg16(top_view, top_anchors, top_inside_inds, len_bases)

        with tf.variable_scope('loss'):
            # RRN
            top_inds = tf.placeholder(shape=[None], dtype=tf.int32, name='top_ind')
            top_pos_inds = tf.placeholder(shape=[None], dtype=tf.int32, name='top_pos_ind')
            top_labels = tf.placeholder(shape=[None], dtype=tf.int32, name='top_label')
            top_targets = tf.placeholder(shape=[None, 4], dtype=tf.float32, name='top_target')
            top_cls_loss, top_reg_loss, pos_top_cls_loss, neg_top_cls_loss = rpn_loss(top_scores, top_deltas, top_inds, top_pos_inds,
                                                  top_labels, top_targets)


    with tf.variable_scope(imfeature_net_name) as scope:
        if cfg.RGB_BASENET =='resnet':
            rgb_features, rgb_stride= rgb_feature_net_r(rgb_images)
        elif cfg.RGB_BASENET =='xception':
            #rgb_features, rgb_stride = rgb_feature_net_x(rgb_images)
            rgb_features, rgb_stride = rgb_feature_net_xcep(rgb_images)
        elif cfg.RGB_BASENET =='vgg16':
            # rgb_features, rgb_stride = rgb_feature_net(rgb_images)
            rgb_features, rgb_stride = rgb_feature_net_vgg16(rgb_images)


    #debug roi pooling
    # with tf.variable_scope('after') as scope:
    #     roi_rgb, roi_idxs = tf_roipooling(rgb_images, rgb_rois, 100, 200, 1)
    #     tf.summary.image('roi_rgb',roi_rgb)

    with tf.variable_scope(fusion_net_name) as scope:
        # fuse_output = fusion_net(
        #         ([top_features, top_rois, 6, 6, 1. / top_feature_stride],
        #          [rgb_features, rgb_rois, 6, 6, 1. / rgb_stride],),
        #         num_class, out_shape)

        fuse_output = early_fusion(
            ([top_features, top_rois, 6, 6, 1. / top_feature_stride],
             [rgb_features, rgb_rois, 6, 6, 1. / rgb_stride]))


        # include background class
        with tf.variable_scope('predict') as scope:
            dim = np.product([*out_shape])
            fuse_scores = linear(fuse_output, num_hiddens=num_class, name='score')
            fuse_probs = tf.nn.softmax(fuse_scores, name='prob')
            fuse_deltas = linear(fuse_output, num_hiddens=dim * num_class, name='box')
            fuse_deltas = tf.reshape(fuse_deltas, (-1, num_class, *out_shape))

        with tf.variable_scope('loss') as scope:
            fuse_labels = tf.placeholder(shape=[None], dtype=tf.int32, name='fuse_label')
            fuse_targets = tf.placeholder(shape=[None, *out_shape], dtype=tf.float32, name='fuse_target')
            fuse_cls_loss, fuse_reg_loss, pos_fuse_cls_loss, neg_fuse_cls_loss = fuse_loss(fuse_scores, fuse_deltas, fuse_labels, fuse_targets)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    return {
        'top_anchors':top_anchors,
        'top_inside_inds':top_inside_inds,
        'top_view':top_view,
        'rgb_images':rgb_images,
        'top_rois':top_rois,
        'rgb_rois': rgb_rois,

        'top_cls_loss': top_cls_loss,
        'top_reg_loss': top_reg_loss,
        'fuse_cls_loss': fuse_cls_loss,
        'fuse_reg_loss': fuse_reg_loss,

        'top_features': top_features,
        'top_scores': top_scores,
        'top_probs': top_probs,
        'top_deltas': top_deltas,
        'train_proposals': train_proposals,
        'train_proposal_scores': train_proposal_scores,
        'infer_proposals': infer_proposals,
        'infer_proposal_scores': infer_proposal_scores,

        'top_inds': top_inds,
        'top_pos_inds':top_pos_inds,

        'top_labels':top_labels,
        'top_targets' :top_targets,

        'fuse_labels':fuse_labels,
        'fuse_targets':fuse_targets,

        'fuse_probs':fuse_probs,
        'fuse_scores':fuse_scores,
        'fuse_deltas':fuse_deltas,

        'top_feature_stride':top_anchors_stride,

        'model_updates': update_ops,

        'pos_top_cls_loss': pos_top_cls_loss,
        'neg_top_cls_loss': neg_top_cls_loss,
        'pos_fuse_cls_loss': pos_fuse_cls_loss,
        'neg_fuse_cls_loss': neg_fuse_cls_loss

    }


if __name__ == '__main__':
    import  numpy as np
    x =tf.placeholder(tf.float32,(None),name='x')
    y = tf.placeholder(tf.float32,(None),name='y')
    idxs = tf.where(tf.not_equal(x,0))
    # weights = tf.cast(tf.not_equal(x,0),tf.float32)
    y_w = tf.gather(y,idxs)
    sess = tf.Session()
    with sess.as_default():
        ret= sess.run(y_w, feed_dict={
            x:np.array([1.0,1.0,0.,2.]),
            y: np.array([1., 2., 2., 3.]),
        })
        print(ret)
