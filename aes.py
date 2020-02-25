# -*- coding: utf-8 -*-
"""AES.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1i7Z1IHULlsn-zp4XoG-uHMPyb1iiwYO3
"""

from PIL import Image
!pip install pycryptodome 
from Crypto.Cipher import AES
from IPython.display import Image
#from keras.datasets import cifar10

import random
import string
from io import BytesIO
from google.colab import files
uploaded = files.upload()
im = Image.open(BytesIO(uploaded['test.bmp']))

#from google.colab import files
#uploaded = files.upload()

#随机生成16个由小写字母组成的字符串
def key_generator(size = 16, chars = string.ascii_lowercase):
    return ''.join(random.choice(chars) for _ in range(size))




format = "BMP"
#利用函数随机生成一个由小写字母组成的字符串
key = key_generator(16)


# AES加密的明文空间为16的整数倍，不能整除，则需要进行填充
#在对应的ascii中，"\x00"表示为0x00，具体的值为NULL，b表示以字节表示
def pad(data):
    return data + b"\x00" * (16 - len(data) % 16)


# 将图像的数据映射为RGB
def trans_format_RGB(data):
    #tuple:不可变,保证数据不丢失
    red, green, blue = tuple(map(lambda e: [data[i] for i in range(0, len(data)) if i % 3 == e], [0, 1, 2]))
    pixels = tuple(zip(red, green, blue))
    return pixels


def encrypt_image_ecb(filename):
    #打开bmp图片，然后将之转换为RGB图像
    im = Image.open(filename)
    #将图像数据转换为像素值字节
    value_vector = im.convert("RGB").tobytes()

    imlength = len(value_vector)
    #for i in range(original):
        #print(data[i])
    #将填充、加密后的数据进行像素值映射
    value_encrypt = trans_format_RGB(aes_ecb_encrypt(key, pad(value_vector))[:imlength])
    #for i in range(original):
        #print(new[i])

    #创建一个新对象，存储相对应的值
    im2 = Image.new(im.mode, im.size)
    im2.putdata(value_encrypt)

    # 将对象保存为对应格式的图像
    im2.save(filename_encrypted_ecb + "." + format, format)

def encrypt_image_cbc(filename):
    #打开bmp图片，然后将之转换为RGB图像
    im = Image.open(filename)
    value_vector = im.convert("RGB").tobytes()

    # 将图像数据转换为像素值字节
    imlength = len(value_vector)

    # 将填充、加密后的数据进行像素值映射
    value_encrypt = trans_format_RGB(aes_cbc_encrypt(key, pad(value_vector))[:imlength])

    # 创建一个新对象，存储相对应的值
    im2 = Image.new(im.mode, im.size)
    im2.putdata(value_encrypt)

    # 将对象保存为对应格式的图像
    im2.save(filename_encrypted_cbc + "." + format, format)

# CBC加密
def aes_cbc_encrypt(key, data, mode=AES.MODE_CBC):
    #IV为随机值
    IV = key_generator(16)
    aes = AES.new(key, mode, IV)
    new_data = aes.encrypt(data)
    return new_data


# ECB加密
def aes_ecb_encrypt(key, data, mode=AES.MODE_ECB):
    #默认模式为ECB加密
    aes = AES.new(key, mode)
    new_data = aes.encrypt(data)
    return new_data



encrypt_image_ecb(filename)
encrypt_image_cbc(filename)
