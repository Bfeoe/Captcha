import os
import random
from PIL import Image
from captcha.image import ImageCaptcha
import torch

String = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

# 图片宽度，图片高度，字符长度，字体大小
# 我把图像调小了一点
img_width = 160
img_height = 60
CHAR_NUMBER = 4
fontsize = [42, 50, 56]


# 随机生成一个四位的字符串
def random_captcha_text():
    captcha_text = []
    for i in range(CHAR_NUMBER):
        c = random.choice(String)
        captcha_text.append(c)
    # join()方法将list中的元素组合成所需的字符串
    return ''.join(captcha_text)


# 生成字符对应的验证码图片和图片的标题
def gen_captcha_text_and_image():
    image = ImageCaptcha(width=img_width, height=img_height, font_sizes=fontsize)
    captcha_text = random_captcha_text()
    # 生成chars内容的验证码图片的Image对象
    captcha_image = Image.open(image.generate(captcha_text))
    return captcha_text, captcha_image


# 还可以增加更多的干扰点和干扰曲线，我暂时先不操作了
# create_noise_dots(image, color, width=3, number=30) 生成验证码干扰点
# create_noise_curve(image, color) 生成验证码干扰曲线


def main(path, count):
    print('save path:', path)
    if not os.path.exists(path):
        os.makedirs(path)
    for i in range(count):
        text, image = gen_captcha_text_and_image()
        filename = text + '.png'
        image.save(path + os.path.sep + filename)
        print('saved %d : %s' % (i + 1, filename))


# 测试集
# 同清姐的数据集数量，看看效果
# if __name__ == '__main__':
#     main('./data/train', 30000)
#     main('./data/test', 5000)


# 将字符转为独热编码
def one_hot_encode(label):
    cols = len(String)
    rows = CHAR_NUMBER
    result = torch.zeros((rows, cols), dtype=float)
    for i, char in enumerate(label):
        j = String.index(char)
        result[i, j] = 1.0

    return result.view(1, -1)[0]


# 试试
# print(one_hot_encode("0W5v"))


# 将独热编码转为字符
def one_hot_decode(pred_result):
    pred_result = pred_result.view(-1, len(String))
    index_list = torch.argmax(pred_result, dim=1)
    text = "".join([String[i] for i in index_list])
    return text


# 试试
# print(one_hot_decode(one_hot_encode("1d5a")))
