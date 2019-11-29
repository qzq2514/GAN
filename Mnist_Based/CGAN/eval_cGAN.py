import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import time

# cGAN_mnist_noLabel-300.pb:对生成器的图像固定都使用0标签,
# 那么在测试的时候无论使用什么标签作为条件,都只会生成0图片
#models/pb/cGAN_mnist-1600.pb
#models_fashion/pb/cGAN_fashion-1100.pb
model_path = 'models_fashion/pb/cGAN_fashion-1200.pb'

image_height = 28
image_width = 28
image_flatten_size = image_height*image_width
prior_size=100
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
    label_place = sess.graph.get_tensor_by_name('label:0')
    ind = 0
    while True:
        #指定标签
        label_org = np.array([[9]])
        #随机标签
        # label_org = np.random.randint(0,10,size=[1,1])

        label = one_hot[label_org].squeeze(axis=1)
        z_prior = np.random.normal(0, 1, size=(1, prior_size))
        image_output = sess.run(generated_output,feed_dict={
                                prior_input:z_prior,label_place:label})

        image_reshape_org = image_output[0].reshape((image_height, image_width))

        #因为输出的元素值范围在-1~1之,所以正确的应该是先加1除以2再乘以255
        #上面直接乘以255虽然也能达到效果,但是不准确
        image_reshape = ((image_reshape_org + 1) / 2) * 255.0
        image_show = image_reshape.astype(np.uint8)

        print("label:", label_org[0, 0])
        cv2.imwrite("result_fashion/{}_{}.jpg".format(label_org[0, 0],ind),image_show)
        image_show = cv2.resize(image_show, (image_height * 2, image_width * 2))
        cv2.imshow("image_fine", image_show)
        cv2.waitKey(0)
        ind+=1



if __name__ == '__main__':
    eval()