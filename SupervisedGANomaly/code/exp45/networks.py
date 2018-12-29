#-----------------------------------------------------------------
# Function: Define network architectures used for defect detection
# Author:   Changzhi Luo
# Date:     20180908
#-----------------------------------------------------------------
import tensorflow as tf
import tensorflow.contrib.slim as slim

# Define generator
def gnet(input_data): #256 * 448 -> 4 * 7 (after 5 conv)
    #print("input_data shape is:", input_data.shape)
    with tf.variable_scope('gnet_scope', reuse=tf.AUTO_REUSE) as scope:
        # Encoder1
        with slim.arg_scope([slim.conv2d, slim.fully_connected], stride = 2, padding = 'SAME', \
                            weights_initializer = tf.truncated_normal_initializer(0.0, 0.01), \
                            weights_regularizer = slim.l2_regularizer(0.0005)):
            #downsampling
            conv1 = slim.conv2d(input_data, 32, [4, 4], scope = 'conv1')  # 128 * 128 * 32
            bn1 = slim.batch_norm(conv1, decay = 0.9, epsilon = 1e-5, scope = 'bn1') 
            relu1 = tf.nn.leaky_relu(bn1)

            conv2 = slim.conv2d(relu1, 64, [4, 4], scope = 'conv2')        # 64 * 64 * 64
            bn2 = slim.batch_norm(conv2, decay = 0.9, epsilon = 1e-5, scope = 'bn2') 
            relu2 = tf.nn.leaky_relu(bn2)

            conv3 = slim.conv2d(relu2, 128, [4, 4], scope = 'conv3')        # 32 * 32 * 128
            bn3 = slim.batch_norm(conv3, decay = 0.9, epsilon = 1e-5, scope = 'bn3') 
            relu3 = tf.nn.leaky_relu(bn3)

            conv4 = slim.conv2d(relu3, 256, [4, 4], scope = 'conv4')        # 16 * 16 * 256
            bn4 = slim.batch_norm(conv4, decay = 0.9, epsilon = 1e-5, scope = 'bn4') 
            relu4 = tf.nn.leaky_relu(bn4)

            conv5 = slim.conv2d(relu4, 256, [4, 4], scope = 'conv5')        # 8 * 8 * 512
            bn5 = slim.batch_norm(conv5, decay = 0.9, epsilon = 1e-5, scope = 'bn5') 
            relu5 = tf.nn.leaky_relu(bn5)

            conv6 = slim.conv2d(relu5, 256, [4, 4], scope = 'conv6')        # 4 * 4 * 512
            bn6 = slim.batch_norm(conv6, decay = 0.9, epsilon = 1e-5, scope = 'bn6') 
            relu6 = tf.nn.leaky_relu(bn6)
            
            #conv7 = slim.conv2d(relu6, 512, [4, 4], scope = 'conv7')        # 2 * 2 * 512
            #bn7 = slim.batch_norm(conv7, decay = 0.9, epsilon = 1e-5, scope = 'bn7') 
            #relu7 = tf.nn.leaky_relu(bn7)
        
            conv7 = slim.conv2d(relu6, 2048, [4, 4], stride = 1, padding = 'VALID', scope = 'conv7') # 1 * 1 * 2048, use conv as fc
            # bottle1 = tf.tanh(conv7)
            bottle1 = tf.nn.relu(conv7)
        
        # Decoder1
        # 只在第一层加unet连接
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], stride = 2, padding = 'SAME', \
                    weights_initializer = tf.truncated_normal_initializer(0.0, 0.01), \
                    weights_regularizer = slim.l2_regularizer(0.0005)):
            #upsampling
            dconv1 = slim.conv2d_transpose(conv7, 256, [4, 4], stride = 1, padding = 'VALID', scope = 'dconv1') # 4 * 4 * 512
            bn8 = slim.batch_norm(dconv1, decay = 0.9, epsilon = 1e-5, scope = 'bn8')
            relu8 = tf.nn.relu(bn8)

            dconv2 = slim.conv2d_transpose(relu8, 256, [4, 4], scope = 'dconv2') # 8 * 8 * 512
            bn9 = slim.batch_norm(dconv2, decay = 0.9, epsilon = 1e-5, scope = 'bn9') 
            relu9 = tf.nn.relu(bn9)

            dconv3 = slim.conv2d_transpose(relu9, 256, [4, 4], scope = 'dconv3') # 16 * 16 * 256
            bn10 = slim.batch_norm(dconv3, decay = 0.9, epsilon = 1e-5, scope = 'bn10') 
            relu10 = tf.nn.relu(bn10)

            dconv4 = slim.conv2d_transpose(relu10, 128, [4, 4], scope = 'dconv4') # 32 * 32 * 128
            bn11 = slim.batch_norm(dconv4, decay = 0.9, epsilon = 1e-5, scope = 'bn11')
            relu11 = tf.nn.relu(bn11)

            dconv5 = slim.conv2d_transpose(relu11, 64, [4, 4], scope = 'dconv5') # 64 * 64* 64
            bn12 = slim.batch_norm(dconv5, decay = 0.9, epsilon = 1e-5, scope = 'bn12')
            relu12 = tf.nn.relu(bn12)

            dconv6 = slim.conv2d_transpose(relu12, 32, [4, 4], scope = 'dconv6') # 128 * 128 * 32
            bn13 = slim.batch_norm(dconv6, decay = 0.9, epsilon = 1e-5, scope = 'bn13')
            relu13 = tf.nn.relu(bn13)

            ##output reconstructed image
            dconv7 = slim.conv2d_transpose(relu13, 3, [4, 4], scope = 'dconv7') # 256 * 256 * 3
            output_data = tf.tanh(dconv7)

        # Encoder2
        with slim.arg_scope([slim.conv2d, slim.fully_connected], stride = 2, padding = 'SAME', \
                    weights_initializer = tf.truncated_normal_initializer(0.0, 0.01), \
                    weights_regularizer = slim.l2_regularizer(0.0005)):
            #downsampling
            conv1_2 = slim.conv2d(output_data, 32, [4, 4], scope = 'conv1_2')  # 128 * 128 * 32
            bn1_2 = slim.batch_norm(conv1_2, decay = 0.9, epsilon = 1e-5, scope = 'bn1_2') 
            relu1_2 = tf.nn.leaky_relu(bn1_2)

            conv2_2 = slim.conv2d(relu1_2, 64, [4, 4], scope = 'conv2_2')        # 64 * 64 * 64
            bn2_2 = slim.batch_norm(conv2_2, decay = 0.9, epsilon = 1e-5, scope = 'bn2_2') 
            relu2_2 = tf.nn.leaky_relu(bn2_2)

            conv3_2 = slim.conv2d(relu2_2, 128, [4, 4], scope = 'conv3_2')        # 32 * 32 * 128
            bn3_2 = slim.batch_norm(conv3_2, decay = 0.9, epsilon = 1e-5, scope = 'bn3_2')
            relu3_2 = tf.nn.leaky_relu(bn3_2)
            
            conv4_2 = slim.conv2d(relu3_2, 256, [4, 4], scope = 'conv4_2')        # 16 * 16 * 256
            bn4_2 = slim.batch_norm(conv4_2, decay = 0.9, epsilon = 1e-5, scope = 'bn4_2') 
            relu4_2 = tf.nn.leaky_relu(bn4_2)

            conv5_2 = slim.conv2d(relu4_2, 256, [4, 4], scope = 'conv5_2')        # 8 * 8 * 512
            bn5_2 = slim.batch_norm(conv5_2, decay = 0.9, epsilon = 1e-5, scope = 'bn5_2') 
            relu5_2 = tf.nn.leaky_relu(bn5_2)

            conv6_2 = slim.conv2d(relu5_2, 256, [4, 4], scope = 'conv6_2')        # 4 * 4 * 512
            bn6_2 = slim.batch_norm(conv6_2, decay = 0.9, epsilon = 1e-5, scope = 'bn6_2') 
            relu6_2 = tf.nn.leaky_relu(bn6_2)
        
            conv7_2 = slim.conv2d(relu6_2, 2048, [4, 4], stride = 1, padding = 'VALID', scope = 'conv7_2') # 1 * 1 * 1024, use conv as fc
            # bottle2 = tf.tanh(conv7_2)
            bottle2 = tf.nn.relu(conv7_2)

    return bottle1, bottle2, output_data
    
# Decoder3 
def cnet(input1, input2):  # inputs are conv7 and conv7_2
    # Actually perform fc here, and activation is no need.
    with tf.variable_scope('cnet_scope') as scope: 
        with slim.arg_scope([slim.conv2d], \
                    activation_fn = tf.nn.relu, \
                    weights_initializer = tf.truncated_normal_initializer(0.0, 0.01), \
                    weights_regularizer = slim.l2_regularizer(0.0005)):
            conv1_bottle = tf.subtract(input1, input2) # 1 * 1 * 512
            conv2_bottle = slim.conv2d(conv1_bottle, 2048, [1, 1], stride = 1, padding = 'VALID', scope = 'conv2_bottle') # 1 * 1 * 512
            # drop1_bottle = slim.dropout(conv2_bottle, 0.5, scope = 'drop1_bottle')
            conv3_bottle = slim.conv2d(conv2_bottle, 2048, [1, 1], stride = 1, padding = 'VALID', scope = 'conv3_bottle') # 1 * 1 * 256
            # drop2_bottle = slim.dropout(conv3_bottle, 0.5, scope = 'drop2_bottle')
            conv3_bottle_norm = tf.tanh(conv3_bottle)
            conv4_bottle = slim.conv2d(conv3_bottle_norm, 2, [1, 1], stride = 1, padding = 'VALID', scope = 'conv4_bottle') # 2 * 1 * 1
            #conv4_bottle_norm = tf.tanh(conv4_bottle) #使用hinge loss不能将值限定在0~1，因此不宜用tanh
            #cls_score = tf.reduce_mean(conv4_bottle_norm, [2, 3]) # classification score (0 for NG, and 1 for OK) 
            cls_score = tf.reduce_mean(conv4_bottle, [1, 2]) # classification score (0 for NG, and 1 for OK) , reduce h and w
    return cls_score

# Define discriminator, Decoder2
def dnet(input_data): #256 * 448 -> 16 * 28 (after 5 conv)
    #print("input_data shape is:", input_data.shape)
    #with tf.variable_scope('dnet_scope', reuse = False) as scope:
    # 更新G Net时会用到这里的参数，但是不能改为True，因为更新D时不能reuse，因此要改为tf.AUTO_REUSE
    with tf.variable_scope('dnet_scope', reuse=tf.AUTO_REUSE) as scope: 
    # with tf.variable_scope('dnet_scope', reuse = tf.AUTO_REUSE) as scope: # reuse有空格都是错的！
        with slim.arg_scope([slim.conv2d], stride = 2, padding = 'SAME', \
                            weights_initializer = tf.truncated_normal_initializer(0.0, 0.01), \
                            weights_regularizer = slim.l2_regularizer(0.0005)):
            #downsampling
            conv1_d = slim.conv2d(input_data, 32, [4, 4], scope = 'conv1_d')  # 128 * 128 * 32
            bn1_d = slim.batch_norm(conv1_d, decay = 0.9, epsilon = 1e-5, scope = 'bn1_d') 
            relu1_d = tf.nn.relu(bn1_d)

            conv2_d = slim.conv2d(relu1_d, 64, [4, 4], scope = 'conv2_d')        # 64 * 64 * 64
            bn2_d = slim.batch_norm(conv2_d, decay = 0.9, epsilon = 1e-5, scope = 'bn2_d') 
            relu2_d = tf.nn.relu(bn2_d)

            conv3_d = slim.conv2d(relu2_d, 128, [4, 4], scope = 'conv3_d')        # 32 * 32 * 128
            bn3_d = slim.batch_norm(conv3_d, decay = 0.9, epsilon = 1e-5, scope = 'bn3_d') 
            relu3_d = tf.nn.relu(bn3_d)

            conv4_d = slim.conv2d(relu3_d, 256, [4, 4], scope = 'conv4_d')        # 16 * 16 * 256
            bn4_d = slim.batch_norm(conv4_d, decay = 0.9, epsilon = 1e-5, scope = 'bn4_d') 
            relu4_d = tf.nn.relu(bn4_d)

            conv5_d = slim.conv2d(relu4_d, 256, [4, 4], scope = 'conv5_d')        # 8 * 8 * 512
            bn5_d = slim.batch_norm(conv5_d, decay = 0.9, epsilon = 1e-5, scope = 'bn5_d') 
            relu5_d = tf.nn.relu(bn5_d)

            conv6_d = slim.conv2d(relu5_d, 256, [4, 4], scope = 'conv6_d')        # 4 * 4 * 512
            bn6_d = slim.batch_norm(conv6_d, decay = 0.9, epsilon = 1e-5, scope = 'bn6_d') 
            relu6_d = tf.nn.relu(bn6_d)
            
            #conv7_d = slim.conv2d(relu6_d, 512, [4, 4], scope = 'conv7_d')        # 2 * 2 * 512
            #bn7_d = slim.batch_norm(conv7_d, decay = 0.9, epsilon = 1e-5, scope = 'bn7_d')
            #relu7_d = tf.nn.relu(bn7_d) 

            conv8_d = slim.conv2d(relu6_d, 2048, [4, 4], stride = 1, padding = 'VALID',scope = 'conv8_d')   # 1 * 1 * 2048
            conv8_d_norm = tf.tanh(conv8_d) 

            conv9_d = slim.conv2d(conv8_d_norm, 1, [1, 1], stride = 1, padding = 'VALID', scope = 'conv9_d')   # 1 * 1 * 1
            conv9_d_norm= tf.tanh(conv9_d) # 0~1, the output should be normalized to 0~1

            output_data = tf.reduce_mean(conv9_d_norm, [1, 2, 3]) # N*1*1*1 -> N*1 # conv order NCHW

    return output_data
