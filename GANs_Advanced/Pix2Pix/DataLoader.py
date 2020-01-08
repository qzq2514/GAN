from scipy import misc
import matplotlib.pyplot as plt

import numpy as np
import os


class Pix2Pix_loader:
    def __init__(self, root, image_height, image_width, batch_size, global_step=0):
        self.train_fileList = self.get_filePath(os.path.join(root, "train"))
        self.val_fileList = self.get_filePath(os.path.join(root, "val"))

        self.image_height = image_height
        self.image_width = image_width
        self.batch_size = batch_size

        self.sample_num = len(self.train_fileList)
        self.test_sample_num = len(self.val_fileList)
        self.step_per_epoch = self.sample_num // self.batch_size
        self.batch_id = global_step % self.step_per_epoch

    def get_filePath(self, dir):
        file_paths = [os.path.join(dir, file_name) for file_name in os.listdir(dir)]
        return np.array(file_paths)

    def shuffle_train(self):
        shuffle_ind = np.arange(0, self.sample_num)
        np.random.shuffle(shuffle_ind)
        self.train_fileList = self.train_fileList[shuffle_ind]

    def next_batch_core(self, batch_fileList):
        images_A = []
        images_B = []
        # 不要使用cv2.imread和cv2.resize不然读取和resize后的边缘图会产生轮廓断点的现象
        # 严重影响最终图像生成效果
        for file_path in batch_fileList:
            image = misc.imread(file_path)
            h, w = image.shape[:2]
            image_A = misc.imresize(image[:, :w // 2, ...], [self.image_height, self.image_width])
            image_B = misc.imresize(image[:, w // 2:, ...], [self.image_height, self.image_width])
            images_A.append(image_A)
            images_B.append(image_B)

        #pix2pix需要成对的数据,所以这里不要shuffle
        images_A = np.array(images_A)
        images_B = np.array(images_B)
        return np.array(images_A), np.array(images_B)

    def random_next_test_batch(self):
        indices = np.random.choice(self.test_sample_num, self.batch_size)
        batch_fileList = self.val_fileList[indices]
        return self.next_batch_core(batch_fileList)

    def random_next_train_batch(self):
        indices = np.random.choice(self.sample_num, self.batch_size)
        batch_fileList = self.train_fileList[indices]
        return self.next_batch_core(batch_fileList)

    def next_batch(self):
        if self.batch_id >= self.step_per_epoch:
            self.batch_id = 0
            self.shuffle_train()
        bacth_ind_start = self.batch_id * self.batch_size
        bacth_ind_end = (self.batch_id + 1) * self.batch_size
        batch_fileList = self.train_fileList[bacth_ind_start:bacth_ind_end]
        self.batch_id += 1
        return self.next_batch_core(batch_fileList)

if __name__ == '__main__':
    image_dir="D:/forTensorflow/forGAN/edges2shoes/"
    dataLoader = Pix2Pix_loader(image_dir, 64, 64,batch_size=16,global_step=0)
    while True:
        images_A,images_B= dataLoader.random_next_train_batch()
        fig =plt.figure()
        ax1 = fig.add_subplot(121)
        ax1.imshow(np.uint8(images_A[0]))
        ax2 = fig.add_subplot(122)
        ax2.imshow(np.uint8(images_B[0]))
        # misc.imsave("image_A.jpg", images_A[0])
        # misc.imsave("image_B.jpg", images_B[0])
        plt.show()