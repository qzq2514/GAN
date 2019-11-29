import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import time

model_path = 'models/pb/DCGAN_mnist-1700.pb'

image_height = 28
image_width = 28
prior_size=100
one_hot=np.eye(10)
test_sample_num=50

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
    label_place = sess.graph.get_tensor_by_name('label:0')
    while True:
        #指定标签
        # label = np.array([3])
        #随机标签
        label = np.random.randint(0,10,size=[1,])

        z_prior = np.random.normal(0, 1, size=(1,prior_size))

        # 用于和带有latent code 的infoGAN进行比较
        # 虽然这里输入噪声某个维度发生变化,DCGAN的输出肯定会有变化,但是变化几乎不明显,
        # 且并无几何意义(如字体的粗细、旋转角度等)
        # 而infoGAN带有latent code,可以显式的挖掘数据中的潜在维度的信息,改变latent code某个维度的值
        # 可以明显且有方向地引导生成数据中的某些特征维度的变化
        dims = np.linspace(-1.0, 1.0, test_sample_num)
        for dim in dims:
            z_prior[0,0]=dim

            image_output = sess.run(generated_output,feed_dict={
                                    prior_input:z_prior,label_place:label,
                                    is_training:False})
            image_reshape_org = image_output[0].reshape((image_height,image_width))


            image_reshape = ((image_reshape_org+1)/2)*255.0
            image_show = image_reshape.astype(np.uint8)

            print("label:",label[0])
            image_show=cv2.resize(image_show,(image_height*2,image_width*2))
            cv2.imshow("image_fine", image_show)
            cv2.waitKey(0)



if __name__ == '__main__':
    eval()