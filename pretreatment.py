# 暂时停用，等全部完成之后有空的话再看看bug在哪里，预处理部分见"Noise_Test/test.py"
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os


# 把一个rgb的图转换成一个二值图
def binarization(path):
    img = Image.open(path)
    # 把图像转化成一个灰度图
    img_gray = img.convert("L")
    # 把灰度图组装成数组形式
    img_gray = np.array(img_gray)
    # 得到灰度图的宽和高
    w, h = img_gray.shape
    for x in range(w):
        for y in range(h):
            # 得到每一个像素块里的灰度值
            gray = img_gray[x, y]
            # 如果灰度值小于等于220， 就把它变成黑色
            if gray <= 220:
                img_gray[x, y] = 0
            # 如果灰度值大于220，就把它变成白色
            else:
                img_gray[x, y] = 1

    # plt.figure("", figsize=((w, h)))
    # plt.imshow(img_gray, cmap="gray")
    # plt.axis("off")
    # plt.show()

    return img_gray


# 降噪，也就是处理离群点，如果一个像素点周围只有小于4个黑点的时候，那么这个点就是离群点
def noiseReduction(img_gray, label):
    height, width = img_gray.shape
    for x in range(height):
        for y in range(width):
            cnt = 0
            # 白色的点不用管
            if img_gray[x, y] == 1:
                continue
            else:
                try:
                    if img_gray[x - 1, y - 1] == 0:
                        cnt += 1
                except:
                    pass

                try:
                    if img_gray[x - 1, y] == 0:
                        cnt += 1
                except:
                    pass

                try:
                    if img_gray[x - 1, y + 1] == 0:
                        cnt += 1
                except:
                    pass

                try:
                    if img_gray[x, y - 1] == 0:
                        cnt += 1
                except:
                    pass

                try:
                    if img_gray[x, y + 1] == 0:
                        cnt += 1
                except:
                    pass

                try:
                    if img_gray[x + 1, y - 1] == 0:
                        cnt += 1
                except:
                    pass

                try:
                    if img_gray[x + 1, y] == 0:
                        cnt += 1
                except:
                    pass

                try:
                    if img_gray[x + 1, y + 1] == 0:
                        cnt += 1
                except:
                    pass

                if cnt < 4:  # 周围少于4个点就算是噪点
                    img_gray[x, y] = 1

    plt.figure(" ")
    plt.imshow(img_gray, cmap="gray")
    plt.axis("off")
    plt.savefig("".join(["./data/clean_data", label, ".png"]))


# 把所有的图像都转化成二值图
def image_clean(path):
    captchas = os.listdir("".join([path]))
    for captcha in captchas:
        label = captcha.split(".")[0]
        image_path = "".join([path, captcha])
        im = binarization(image_path)
        noiseReduction(im, label)
        print('change : %s' % label)


# 转换训练集
if __name__ == '__main__':
    image_clean("./data/train/")
