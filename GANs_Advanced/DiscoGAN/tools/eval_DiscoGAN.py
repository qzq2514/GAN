import numpy as np
import tensorflow as tf
from DataLoader import Pix2Pix_loader
import cv2

model_path = 'models/pb/frozen_model.pb'
image_dir = "D:/forTensorflow/forGAN/edges2shoes/"
batch_size = 1
image_height = 64
image_width = 64

def eval():
    sess = tf.Session()
    with tf.gfile.FastGFile(model_path, "rb") as fr:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(fr.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name="")

    sess.run(tf.global_variables_initializer())

    input_A_place = sess.graph.get_tensor_by_name('input_A:0')
    input_B_place = sess.graph.get_tensor_by_name('input_B:0')

    A2B_output = sess.graph.get_tensor_by_name('A2B_output:0')
    B2A_output = sess.graph.get_tensor_by_name('B2A_output:0')

    is_training = sess.graph.get_tensor_by_name('is_training:0')

    dataLoader = Pix2Pix_loader(image_dir, image_height, image_width, batch_size=batch_size)

    while True:
        images_A, images_B = dataLoader.random_next_test_batch()

        _A2B_output = sess.run(A2B_output, feed_dict={input_A_place: images_A,is_training:False})
        _A2B_output = (_A2B_output + 1) / 2 * 255.0
        _A2B_output = _A2B_output.astype(np.uint8)

        _B2A_output = sess.run(B2A_output, feed_dict={input_B_place: images_B,is_training:False})
        _B2A_output = (_B2A_output + 1) / 2 * 255.0
        _B2A_output = _B2A_output.astype(np.uint8)

        cv2.imshow("A", images_A[0])
        cv2.imshow("B", images_B[0])
        cv2.imshow("A2B_output",_A2B_output[0])
        cv2.imshow("B2A_output", _B2A_output[0])
        cv2.waitKey(0)

if __name__ == '__main__':
    eval()