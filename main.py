import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


NUM_CLASSES = 2
IMAGE_SHAPE = (160, 576)
EPOCHS = 40
BATCH_SIZE = 16
DROPOUT = 0.75
LEARNING_RATE = 0.001


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    # load the model and weights
    model = tf.saved_model.loader.load(sess, ['vgg16'], vgg_path)

    # Get Tensors to be returned from graph
    graph = tf.get_default_graph()
    image_input = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return image_input, keep_prob, layer3_out, layer4_out, layer7_out

tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """

    # Apply 1x1 convolution in place of fully connected layer
    conv_1x1 = tf.layers.conv2d(
        vgg_layer7_out, filters=num_classes,
        kernel_size=(1, 1), strides=(1, 1), padding='same',
        kernel_initializer= tf.random_normal_initializer(stddev=0.01),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
    )

    # Upsample fcn8 with size depth=(4096?) to match size of layer 4 so that we can add skip connection with 4th layer
    deconv1 = tf.layers.conv2d_transpose(
        conv_1x1,
        filters=vgg_layer4_out.get_shape().as_list()[-1],
        kernel_size=(4, 4), strides=(2, 2), padding='same',
        kernel_initializer= tf.random_normal_initializer(stddev=0.01),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
    )

    deconv1_plus_vgg_layer4 = tf.add(deconv1, vgg_layer4_out)

    deconv2 = tf.layers.conv2d_transpose(
        deconv1_plus_vgg_layer4,
        filters=vgg_layer3_out.get_shape().as_list()[-1],
        kernel_size=(4, 4), strides=(2, 2), padding='same',
        kernel_initializer= tf.random_normal_initializer(stddev=0.01),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
    )

    deconv2_plus_vgg_layer3 = tf.add(deconv2, vgg_layer3_out)

    deconv3 = tf.layers.conv2d_transpose(
        deconv2_plus_vgg_layer3,
        filters=num_classes,
        kernel_size=(16, 16), strides=(8, 8), padding='same',
        kernel_initializer= tf.random_normal_initializer(stddev=0.01),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
    )

    return deconv3

tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
     # Reshape 4D tensors to 2D, each row represents a pixel, each column a class
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    correct_label_reshaped = tf.reshape(correct_label, (-1, num_classes))

    # Calculate distance from actual labels using cross entropy
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label_reshaped)

    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

    # Take mean for total loss
    loss = tf.reduce_mean(cross_entropy) + tf.reduce_sum(reg_losses)

    # The model implements this operation to find the weights/parameters that would yield correct pixel labels
    train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    return logits, train, loss

tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    for epoch in range(epochs):
        # Create function to get batches
        total_loss = 0
        for X_batch, gt_batch in get_batches_fn(batch_size):
            loss, _ = sess.run(
                [cross_entropy_loss, train_op],
                feed_dict={
                    input_image: X_batch,
                    correct_label: gt_batch,
                    keep_prob: DROPOUT,
                    learning_rate: LEARNING_RATE
                })

            total_loss += loss;

        print("EPOCH {} ...".format(epoch + 1))
        print("Loss = {:.3f}".format(total_loss))
        print()

tests.test_train_nn(train_nn)


def run():
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    # Create function to get batches
    get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), IMAGE_SHAPE)

    # Path to vgg model
    vgg_path = os.path.join(data_dir, 'vgg')

    with tf.Session() as sess:

        # Returns the three layers, keep probability and input layer from the vgg architecture
        image_input, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)

        # The resulting network architecture from adding a decoder on top of the given vgg model
        model_output = layers(layer3_out, layer4_out, layer7_out, NUM_CLASSES)

        # placeholders
        correct_label = tf.placeholder(tf.float32, [None, IMAGE_SHAPE[0], IMAGE_SHAPE[1], NUM_CLASSES])
        learning_rate = tf.placeholder(tf.float32)

        # Returns the output logits, training operation and cost operation to be used
        # - logits: each row represents a pixel, each column a class
        # - train_op: function used to get the right parameters to the model to correctly label the pixels
        # - cross_entropy_loss: function outputting the cost which we are minimizing, lower cost should yield higher accuracy
        logits, train_op, cross_entropy_loss = optimize(model_output, correct_label, learning_rate, NUM_CLASSES)

        # Initialize all variables
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        print("Model build successful, starting training")

        # Train the neural network
        train_nn(
            sess, EPOCHS, BATCH_SIZE, get_batches_fn,
            train_op, cross_entropy_loss, image_input,
            correct_label, keep_prob, learning_rate
        )

        saver = tf.train.Saver()
        save_path = saver.save(sess, "./checkpoint/model.ckpt")
        print("Model saved in path: {}".format(save_path))

        # Run the model with the test images and save each painted output image (roads painted green)
        helper.save_inference_samples(runs_dir, data_dir, sess, IMAGE_SHAPE, logits, keep_prob, image_input)

        print("All done!")


        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
