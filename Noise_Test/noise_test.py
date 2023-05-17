# 图像去噪平滑滤波
# 我想使用opencv的自带函数，与我自编写的作比较，莫名有bug，我也不知道怎么解决
# 使用中值滤波，平均滤波，高斯滤波，方框滤波，以及膨胀腐蚀算法
import cv2
import matplotlib.pyplot as plt

if __name__ == "__main__":
    image = cv2.imread('064D.png')

    plt.subplot(2, 3, 1)
    plt.imshow(image)
    plt.axis('off')
    plt.title('Original')

    # 均值滤波
    result1 = cv2.blur(image, (5, 5))
    plt.subplot(2, 3, 2)
    plt.imshow(result1)
    plt.axis('off')
    plt.title('mean')

    # 方框滤波
    result2 = cv2.boxFilter(image, -1, (5, 5), normalize=1)
    plt.subplot(2, 3, 3)
    plt.imshow(result2)
    plt.axis('off')
    plt.title('box')

    # 高斯滤波
    result3 = cv2.GaussianBlur(image, (3, 3), 0)
    plt.subplot(2, 3, 4)
    plt.imshow(result3)
    plt.axis('off')
    plt.title('gaussian')

    # 中值滤波
    result4 = cv2.medianBlur(image, 3)
    plt.subplot(2, 3, 5)
    plt.imshow(result4)
    plt.axis('off')
    plt.title('median')

    # 膨胀腐蚀算法开运算，先腐蚀后膨胀，能够去除孤立的小噪点，图形中的毛刺
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # result5 = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    # plt.subplot(2, 3, 6)
    # plt.imshow(result5)
    # plt.axis('off')
    # plt.title('open')

    # 膨胀腐蚀算法闭运算，先膨胀后腐蚀，可以修复主体图形中的坑坑洼洼，填补小裂缝，使其目标特征更加完备
    result5 = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    plt.subplot(2, 3, 6)
    plt.imshow(result5)
    plt.axis('off')
    plt.title('close')

    plt.show()
