import numpy as np
import tensorflow as tf
from DataLoader import SVHN_loader
import cv2
model_path = 'models/pb/DTN-1800.pb'

batch_size=1
image_height = 32
image_width = 32

def eval():
    sess = tf.Session()
    with tf.gfile.FastGFile(model_path, "rb") as fr:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(fr.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name="")

    sess.run(tf.global_variables_initializer())

    svhn_input = sess.graph.get_tensor_by_name('svhn_image:0')
    generated_output = sess.graph.get_tensor_by_name('generated_output:0')
    is_training = sess.graph.get_tensor_by_name('is_training:0')

    svhn_loader = SVHN_loader("D:/forTensorflow/forGAN/SVHN/", batch_size)
    ind = 515
    while True:
        batch_images, _ = svhn_loader.random_next_train_batch()
        image_output = sess.run(generated_output,feed_dict={
                            svhn_input:batch_images,is_training:False})

        # image_reshape_org = image_output[0].reshape((image_height,image_width))

        image_output = (image_output+1)/2*255.0
        image_show = image_output.astype(np.uint8)

        cv2.imshow("SVNH", batch_images[0]);cv2.imwrite("result/{}_src.jpg".format(ind),batch_images[0])
        cv2.imshow("MNIST", image_show[0]);cv2.imwrite("result/{}_trg.jpg".format(ind),image_show[0])

        ind+=1

        cv2.waitKey(0)



if __name__ == '__main__':
    eval()