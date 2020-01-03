import numpy as np
from scipy import misc
from os import listdir
from os.path import isfile, join


class DBreader:
    def __init__(self, filedir, batch_size, resize=0, labeled=True, color=True, shuffle=True):
        self.color = color
        self.labeled = labeled

        self.batch_size = batch_size
        tmp_filelist = [(filedir + '/' + f) for f in listdir(filedir) if isfile(join(filedir, f))]
        tmp_filelist = np.array(tmp_filelist)

        self.file_len = len(tmp_filelist)

        self.filelist = []
        self.labellist = []
        if self.labeled:
            for i in range(self.file_len):
                splited = (tmp_filelist[i]).split(" ")
                self.filelist.append(splited[0])
                self.labellist.append(splited[1])
        else:
            self.filelist = tmp_filelist

        self.batch_idx = 0
        self.total_batch = int(self.file_len / batch_size)
        self.idx_shuffled = np.arange(self.file_len)
        if shuffle:
            np.random.shuffle(self.idx_shuffled)
        self.resize = resize

        self.filelist = np.array(self.filelist)
        self.labellist = np.array(self.labellist)

    # Method for get the next batch
    def next_batch(self):
        if self.batch_idx == self.total_batch:
            np.random.shuffle(self.idx_shuffled)
            self.batch_idx = 0

        batch = []
        idx_set = self.idx_shuffled[self.batch_idx * self.batch_size:(self.batch_idx + 1) * self.batch_size]
        batch_filelist = self.filelist[idx_set]

        for i in range(self.batch_size):
            im = misc.imread(batch_filelist[i])
            if self.resize != 0:
                im = misc.imresize(im, self.resize)
                if self.color:
                    if im.shape[2] > 3:
                        im = im[:, :, 0:3]
            batch.append(im)

        if self.labeled:
            label = self.labellist[idx_set]
            self.batch_idx += 1
            return np.array(batch).astype(np.float32), np.array(label).astype(np.int32)

        self.batch_idx += 1
        return np.array(batch).astype(np.float32)
