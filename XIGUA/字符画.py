from PIL import Image,ImageFont,ImageDraw
import numpy as np
import argparsem

def ascii_art(file):
    im = Image.open(file)
    # 转变为灰度图
    im = im.convert("L")

    # 下采样
    sample_rate = 0.15

    font = Image
    new_im_size = [int(x * sample_rate) for x in im.size]
    im = im.resize(new_im_size)

    # 将图片转变为数组
    im = np.array(im)

    # 定义字符画中用到的所有字符
    symbols = np.array(list(".-vM"))

    im = (im - im.min() ) / (im.max() - im.min()) * (symbols.size - 1)

    ascii = symbols[im.astype(int)]
    lines = "\n".join(("".join(r) for r in ascii))
    print(lines)


if __name__ == "__main__":
    ascii_art("ying.jpg")
