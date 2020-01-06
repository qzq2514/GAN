import os
import cv2
import sys
import glob
import random
import numpy as np

file_dir=sys.argv[1]
grid_len=int(sys.argv[2])

grid_num_row=8
grid_num_col=6
margin = 2

tabel_len_width=grid_num_col*grid_len*2
tabel_len_height=grid_num_row*grid_len
file_paths=glob.glob(file_dir+"/*_A.jpg")
random.shuffle(file_paths)

imgShow=np.zeros(shape=[tabel_len_height+margin*(grid_num_row-1),
                        tabel_len_width+margin*(grid_num_col-1),3])
random.shuffle(file_paths)

for ind,src_file_path in enumerate(file_paths):
    if ind >=grid_num_row*grid_num_col:
        break
    x=ind%grid_num_col
    y=ind//grid_num_col

    src_x_start = x * (grid_len*2 + margin)
    src_x_end = src_x_start + grid_len
    trg_x_start = src_x_end
    trg_x_end = src_x_end + grid_len

    y_start = y * (grid_len + margin)
    y_end = y_start + grid_len

    trg_image_path = src_file_path.replace("_A.jpg","_A2B.jpg")
    print(trg_image_path)

    src_image = cv2.imread(src_file_path)
    trg_image = cv2.imread(trg_image_path)

    src_image = cv2.resize(src_image,(grid_len,grid_len))
    trg_image = cv2.resize(trg_image, (grid_len, grid_len))
    imgShow[y_start:y_end, src_x_start:src_x_end, :] = src_image
    imgShow[y_start:y_end, trg_x_start:trg_x_end, :] = trg_image

imgShow=np.array(imgShow,np.uint8)
# print(imgShow)
cv2.imwrite(os.path.join(file_dir,"../imageTable_A2B.jpg"),imgShow)
cv2.waitKey(0)