# 我自己编写的预处理效果属实不尽人意，遂放弃，copy清姐嘀（有时间在做修改）
import cv2  # OpenCV是基于C/C++的，”cv”和”cv2”表示的是底层C API和C++API的区别，”cv2”表示使用的是C++API
import numpy as np
import os

Read_path = './data/train/'
Save_path = "./data/clean_data/"

# Read_path = './data/test/'
# Save_path = "./data/test_clean/"

img_list = os.listdir(Read_path)
for i, v in enumerate(img_list):
    img_list[i] = os.path.splitext(v)[0]
for i, v in enumerate(img_list):
    img = cv2.imread(Read_path + v + '.png')
    # 灰度化，转为单通道
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 二值化，参数:(原数组，阈值，最大值)
    thresh, img = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
    # 方法一：opencv 形态学变换,先膨胀后腐蚀，去除小黑点
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones(shape=(3, 3)))
    # 方法二：中值滤波
    # img = cv2.medianBlur(np.array(img),5)
    # 方法三：高斯滤波，滤波核的尺寸必须是奇数
    # img = cv2.GaussianBlur(img,(3,3),1.5)
    cv2.imwrite(Save_path + v + '.png', img)
    print("降噪图片数量：", i + 1, "张")

WIDTH_PIXELS = 160
HEIGHT_PIXELS = 60
DISQUALIFIED = 0

# 黑色像素个数小于1500的图片判定为不合格
if __name__ == '__main__':
    # 取出目录下所有文件
    img_list = os.listdir(Save_path)
    delete_list = []
    for i, v in enumerate(img_list):
        img_list[i] = os.path.splitext(v)[0]
    for i, v in enumerate(img_list):
        img = cv2.imread(Save_path + v + '.png')
        black_count = 0
        white_count = 0
        # 遍历二值图，为0则black+1，否则white+1
        for i in range(0, HEIGHT_PIXELS - 1):
            for j in range(0, WIDTH_PIXELS - 1):
                # print("DEBUG:img[i,j] = "+str(img[i,j]))
                if img[i, j][0] == 0:  # img[i,j]的格式：[255,255,255]
                    black_count += 1
                else:
                    white_count += 1
        # print("DEBUG:白色个数:"+str(white_count)+" 黑色个数:"+str(black_count))
        if black_count <= 1500:
            DISQUALIFIED += 1
            delete_list.append(v)
    # print("DEBUG:delete_list = "+str(delete_list))
    for i, v in enumerate(delete_list):
        os.remove(Save_path + v + '.png')
    print("DISQUALIFIED =" + str(DISQUALIFIED))
