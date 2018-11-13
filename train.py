import tensorflow as tf
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

sys.path.append('.')
import cvae

mnist = input_data.read_data_sets("./data/MNIST_data/", one_hot=True)

def one_hot_encoding(label=None, num=None):
    '''
    :param label: int 0~9
    :param num: Batch size
    :return: one_hot
    '''
    if label == None:
        if num != None:
            one_hot = np.zeros(shape=(num,10))
            random_labels = np.random.randint(10, size=num)
            for i in range(num):
                label = random_labels[i]
                one_hot[i][label] = 1
            return one_hot
        else:
            return None
    else:
        if num != None:
            one_hot = np.zeros(shape=(num, 10))
            for i in range(num):
                one_hot[i, label] = 1
        else:
            one_hot = np.zeros(shape=(10))
            one_hot[label] = 1
        return one_hot

def train(config):
    '''
     SETTING HYPERPARAMETER (DEFAULT)
     '''
    training_epoch = config.training_epoch
    z_dim = config.z_dim
    batch_size = config.batch_size
    n_data = mnist.train.num_examples
    total_batch = int(mnist.train.num_examples / batch_size)
    total_iteration = training_epoch * total_batch

    # Build Network
    CVAE = cvae.CVAE(config)
    CVAE.build()
    # Optimize Network
    CVAE.optimize(config)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    print("Total the number of Data : " + str(n_data))
    print("Total Step per 1 Epoch: {}".format(total_batch))
    print("The number of Iteration: {}".format(total_iteration))

    for epoch in range(training_epoch):
        avg_cost = 0
        avg_recons = 0
        avg_regular = 0
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)

            _cost, _, _recons, _regular = sess.run([CVAE.cost, CVAE.optimizer, CVAE.recons, CVAE.regular],
                                                   feed_dict={CVAE.X: batch_xs, CVAE.Y: batch_ys})
            avg_cost += _cost / total_batch
            avg_recons += _recons / total_batch
            avg_regular += _regular / total_batch

        if epoch % 10 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost),
                  'Recons_Loss =', '{:.9f}'.format(avg_recons),
                  'Regular_Loss =', '{:.9f}'.format(avg_regular))

    print("Training Complete!")

    save_dir = './mode_z_dim_{}'.format(z_dim)
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    save_path = '{}/CVAE.ckpt'.format(save_dir)
    saver.save(sess, save_path)
    print("Saved Model")

    return CVAE, sess

def run_train(config):
    result_dir = './result'
    if not os.path.exists(result_dir): os.makedirs(result_dir)

    '''
    Train 20-dimension model
    '''
    CVAE_20dim, sess = train(config)

    #Just reconstruction image
    n_sample = 8
    test_images, test_labels = mnist.test.next_batch(n_sample)

    reconstruction_images = sess.run(CVAE_20dim.output,
                             feed_dict={CVAE_20dim.X: test_images, CVAE_20dim.Y: test_labels})
    images_list = [test_images, reconstruction_images]

    columns = 8
    rows = 2
    fig, axis = plt.subplots(rows, columns)
    for i in range(columns):
        for j in range(rows):
            axis[j, i].imshow(images_list[j][i].reshape(28, 28))
    plt.savefig('{}/reconstruction.png'.format(result_dir))

    #Generation from Z distribution
    z = np.random.normal(0, 1, size=[n_sample,CVAE_20dim.z_dim])
    # Labels and One_hot_Encoding
    labels = np.array([0,1,2,3,4,9,8,7])
    one_hot_labels = np.zeros(shape=(n_sample, 10))
    for i in range(n_sample):
        one_hot_labels[i][labels[i]] = 1
    generation_images = sess.run(CVAE_20dim.output, feed_dict={CVAE_20dim.sampled_z:z, CVAE_20dim.Y:one_hot_labels})
    images_list = [generation_images]
    columns = 8
    rows = 1
    fig, axis = plt.subplots(rows, columns)
    for i in range(columns):
        for j in range(rows):
            axis[j+i].imshow(images_list[j][i].reshape(28, 28))
    plt.savefig('{}/generation.png'.format(result_dir))
    plt.close()

    tf.reset_default_graph()

    '''
    Train 2-dimension model
    '''
    config.z_dim = 2
    CVAE_2dim, sess = train(config)

    # Manifold 2-dimension
    n_sample = 4000
    test_images, test_labels = mnist.test.next_batch(n_sample)
    z = sess.run(CVAE_2dim.sampled_z, feed_dict={CVAE_2dim.X:test_images,
                                                 CVAE_2dim.Y:test_labels})

    fig = plt.figure()
    plt.scatter(z[:, 0], z[:, 1], c=np.argmax(test_labels, 1))
    plt.colorbar()
    plt.grid()
    plt.savefig('{}/manifold.png'.format(result_dir))
    plt.close(fig)

    # Manifold 2-dimension walking
    label = one_hot_encoding(7, num=1)
    x_space = np.linspace(start=-2, stop=2, num=20)
    y_space = np.linspace(start=-2, stop=2, num=20)
    result_size = [28*20, 28*20]
    result_image = np.empty(shape=result_size)
    interval = 28

    for x_index, x in enumerate(x_space):
        for y_index, labels in enumerate(y_space):
            z = np.expand_dims([x,labels], axis=0)
            generation_images = sess.run(CVAE_2dim.output, feed_dict={CVAE_2dim.sampled_z:z, CVAE_2dim.Y:label})
            generation_images = np.squeeze(generation_images, axis=0)

            height_start = y_index*interval
            height_end = (y_index+1)*interval
            width_start = x_index*interval
            width_end = (x_index+1)*interval

            result_image[height_start:height_end, width_start:width_end] = generation_images.reshape(28,28)

    fig = plt.figure()
    plt.imshow(result_image, cmap="gray")
    plt.savefig('{}/walking.png'.format(result_dir))
    plt.close(fig)