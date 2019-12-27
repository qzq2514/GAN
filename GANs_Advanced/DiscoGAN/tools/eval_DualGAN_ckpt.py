import numpy as np
import tensorflow as tf
from DataLoader import Pix2Pix_loader
import cv2

model_path = 'models/ckpt/DualGAN_64_1227_new.ckpt-10000'
image_dir = "D:/forTensorflow/forGAN/edges2shoes/"
batch_size = 1
image_height = 64
image_width = 64

def eval():
    with tf.Session() as sess:
        ckpt_path = model_path
        saver = tf.train.import_meta_graph(ckpt_path + '.meta')
        saver.restore(sess, ckpt_path)

        input_A_place = tf.get_default_graph().get_tensor_by_name('input_A:0')
        input_B_place = tf.get_default_graph().get_tensor_by_name('input_B:0')
        keep_prob_place = tf.get_default_graph().get_tensor_by_name('keep_prob:0')
        is_training = tf.get_default_graph().get_tensor_by_name('is_training:0')

        A2B_output = tf.get_default_graph().get_tensor_by_name('A2B_output:0')
        B2A_output = tf.get_default_graph().get_tensor_by_name('B2A_output:0')

        dataLoader = Pix2Pix_loader(image_dir, image_height, image_width, batch_size=batch_size)

        index = 1
        while True:
            images_A, images_B = dataLoader.random_next_test_batch()

            _A2B_output = sess.run(A2B_output, feed_dict={input_A_place: images_A,
                                            is_training:False,keep_prob_place:0.5})
            _A2B_output = (_A2B_output + 1) / 2 * 255.0
            _A2B_output = _A2B_output.astype(np.uint8)

            _B2A_output = sess.run(B2A_output, feed_dict={input_B_place: images_B,
                                            is_training:False,keep_prob_place:0.5})
            _B2A_output = (_B2A_output + 1) / 2 * 255.0
            _B2A_output = _B2A_output.astype(np.uint8)

            cv2.imshow("A", images_A[0])
            cv2.imshow("B", images_B[0])
            cv2.imshow("A2B_output",_A2B_output[0])
            cv2.imshow("B2A_output", _B2A_output[0])

            cv2.waitKey(0)

if __name__ == '__main__':
    eval()