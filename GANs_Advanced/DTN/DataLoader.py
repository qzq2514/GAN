import pickle
from scipy.io import loadmat
import scipy.misc
import numpy as np
import cv2
import os

class SVHN_loader:
    def __init__(self,dir,batch_size,kinds=None,one_hot=False):
                            # loadmat(os.path.join(dir,"train_32x32.mat"))
        self.train_data = scipy.io.loadmat(os.path.join(dir,"train_32x32.mat"))
        self.train_labels = self.train_data["y"].squeeze(axis=1)
        self.train_labels[self.train_labels==10] = 0
        self.train_images = self.train_data["X"].transpose([3, 0, 1, 2])

        self.test_data = loadmat(os.path.join(dir, "test_32x32.mat"))
        self.test_labels = self.test_data["y"].squeeze(axis=1)
        self.test_labels[self.test_labels == 10] = 0
        self.test_images = self.test_data["X"].transpose([3, 0, 1, 2])

        self.class_num=len(set(self.train_labels))

        ont_hot_eye=np.eye(self.class_num)
        self.batch_size = batch_size
        self.batch_id = 0

        indices_train = range(len(self.train_labels))
        indices_test = range(len(self.test_labels))

        if kinds is not None:
            indices_train = [ind for ind in indices_train if self.train_labels[ind] in kinds]
            indices_test = [ind for ind in indices_test if self.test_labels[ind] in kinds]

        self.train_images = np.array(self.train_images)[indices_train]
        self.train_labels = np.array(self.train_labels)[indices_train]

        self.test_images = np.array(self.test_images)[indices_test]
        self.test_labels = np.array(self.test_labels)[indices_test]

        if one_hot:
            self.train_labels = ont_hot_eye[self.train_labels]
            self.test_labels = ont_hot_eye[self.test_labels]

        self.sample_num = len(self.train_labels)
        self.test_sample_num = len(self.test_labels)
        self.step_per_epoch = self.sample_num // self.batch_size

    def shuffle_train(self):

        shuffle_ind = np.arange(0, self.sample_num)
        np.random.shuffle(shuffle_ind)
        self.train_images = self.train_images[shuffle_ind]
        self.train_labels = self.train_labels[shuffle_ind]

    def random_next_test_batch(self):

        indices = np.random.choice(self.test_sample_num, self.batch_size)
        batch_images = self.test_images[indices]
        batch_labels = self.test_labels[indices]

        return batch_images,batch_labels

    def random_next_train_batch(self):

        indices = np.random.choice(self.sample_num, self.batch_size)
        batch_images = self.train_images[indices]
        batch_labels = self.train_images[indices]

        return batch_images,batch_labels

    def next_batch(self):
        if self.batch_id>=self.step_per_epoch:
            self.batch_id=0
        bacth_ind_start = self.batch_id * self.batch_size
        bacth_ind_end = (self.batch_id+1) * self.batch_size
        batch_images=self.train_images[bacth_ind_start:bacth_ind_end]
        batch_labels=self.train_labels[bacth_ind_start:bacth_ind_end]
        self.batch_id+=1
        return batch_images,batch_labels


if __name__ == '__main__':
    batch_size = 1280
    svhn=SVHN_loader("D:/forTensorflow/forGAN/SVHN/",batch_size,None,one_hot=False)

    while True:
        batch_images, batch_labels=svhn.next_batch()
        # print(batch_images.shape)
        # print("{}/{}".format(ind+1,step_per_epoch))
        for image,label in zip(batch_images,batch_labels):
            print("label:",label)
            print(np.min(image),np.max(image))
            if label==1:
                cv2.imshow("image", image)
                cv2.waitKey(0)
