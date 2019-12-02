import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import time

# "models/pb/BEGAN_cartoon-5900.pb"
# "models_new/pb/BEGAN_cartoon_new-24100.pb"
model_path = 'models_new/pb/BEGAN_cartoon_new-24000.pb'

image_height = 64
image_width = 64
image_channal = 3
prior_size=150
batch_size=64
one_hot=np.eye(10)

def eval():
    sess = tf.Session()
    with tf.gfile.FastGFile(model_path, "rb") as fr:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(fr.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name="")

    sess.run(tf.global_variables_initializer())

    prior_input = sess.graph.get_tensor_by_name('z_prior:0')
    generated_output = sess.graph.get_tensor_by_name('generated_output:0')
    is_training = sess.graph.get_tensor_by_name('is_training:0')

    ind=100
    while True:
        z_prior = np.random.uniform(-1, 1, size=(batch_size,prior_size))
        image_output = sess.run(generated_output,feed_dict={
                                prior_input:z_prior,
                                is_training:False})
        image_reshape_org = image_output[0].reshape((image_height,image_width,image_channal))
        print(image_reshape_org)
        image_reshape = ((image_reshape_org+1)/2)*255.0
        # image_reshape = image_reshape_org * 255.0
        image_show = image_reshape.astype(np.uint8)

        cv2.imwrite("result_cartoon/{}.jpg".format(ind), image_show)
        image_show=cv2.resize(image_show,(image_height*2,image_width*2))
        cv2.imshow("image_fine", image_show)
        cv2.waitKey(0)
        ind+=1



if __name__ == '__main__':
    eval()