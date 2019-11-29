import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import time

#InfoGAN_mnist_38noise_DC-2000:不使用latent code的损失,也不使用分类损失,就是一般的DCGAN
#InfoGAN_mnist_38noise_1-2000:latent_loss的损失权重为1
#InfoGAN_mnist_38noise_01-2000:latent_loss的损失权重为0.1
#InfoGAN_mnist_100noise_1-2000:latent_loss的损失权重为1,且初始噪声长度为100,以上三者都是38

# 可对比InfoGAN_mnist_38noise_DC和InfoGAN_mnist_38noise_1可以看到,在相同噪声长度的情况下,
# InfoGAN随着latent code变化更明显且有意义,而DCGAN变化不明显(此时前两维的latent code和噪声
# 中其他维度的值没什么区别,只是一般的噪声,不带有潜在编码的意义)

model_path = 'models_fashion/pb/InfoGAN_fashion_38noise_-2000.pb'

image_height = 28
image_width = 28
prior_size=38
test_sample_num = 30
latent_code_size = 2
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
    latent_code_input = sess.graph.get_tensor_by_name('latent_code:0')
    generated_output = sess.graph.get_tensor_by_name('generated_output:0')
    label_place = sess.graph.get_tensor_by_name('label:0')
    while True:
        #指定标签
        # label_org = np.array([[3]])
        #随机标签
        label_org = np.random.randint(0, 10, size=[1, ])
        z_prior = np.random.uniform(-1, 1, size=(1, prior_size))

        latent_codes = np.ones([latent_code_size, test_sample_num])
        latent_codes[1, :] = np.linspace(-1.0, 1.0, test_sample_num)
        latent_codes[0, :] = 1.0
        latent_codes = latent_codes.T

        #noise和label不变,改变latent code,查看对应的变化
        for latent_code in latent_codes:
            latent_code = latent_code[np.newaxis,:]
            image_output = sess.run(generated_output,feed_dict={
                                    prior_input:z_prior,label_place:label_org,
                                    latent_code_input:latent_code})
            # print(image_output)
            image_reshape_org = image_output[0].reshape((image_height,image_width))


            image_reshape = ((image_reshape_org+1)/2)*255.0
            image_show = image_reshape.astype(np.uint8)

            print("label:",label_org)
            image_show=cv2.resize(image_show,(image_height*2,image_width*2))
            cv2.imshow("image_fine", image_show)
            cv2.waitKey(0)



if __name__ == '__main__':
    eval()