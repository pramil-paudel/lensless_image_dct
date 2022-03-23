import numpy as np
from PIL import Image
from matplotlib import image as mpimg
from matplotlib import pyplot as plt
from scipy.fftpack import dct, idct
from scipy.io import loadmat
from skimage.transform import resize
import cv2


import flatcam

# calib = loadmat('flatcam_calibdata.mat')  # load calibration data


# ReconstructØ
lmbd = 3e-4  # L2 regularization parameter


def transfom_to_multiple_images(image_to_transform):
    '''
    :param image_to_transform:
    :param data_path: folder of lensless images from rice university
    :return: Bayers Pattern Images
    '''

    global calib
    input_image = np.asarray(image_to_transform)
    calib = loadmat('flatcam_calibdata.mat')
    '''
    Covert into Bayers Pattern 
    '''
    flatcam.clean_calib(calib)
    bayers_pattern = flatcam.fc2bayer(input_image, calib)
    '''
    Coverting into RGB Pattern (lets try without normalization ) 
    '''
    rgb_image = flatcam.bayer2rgb(bayers_pattern)
    rgb_image_in_255 = rgb_image * 255
    '''
    First resize into (64,64) and take DCT : it will convert into 
    '''
    #rgb_image_64 = resize(rgb_image_in_255, (64, 64)).astype(np.uint8)
    #rgb_image_32 = resize(rgb_image_in_255, (32, 32)).astype(np.uint8)
    # This resize has different output ???
    rgb_image_64 = cv2.resize(rgb_image_in_255, (64, 64), interpolation=cv2.INTER_CUBIC)
    rgb_image_32 = cv2.resize(rgb_image_in_255, (32, 32), interpolation=cv2.INTER_CUBIC)
    # print(rgb_image_64)
    '''
    Converting 64 * 64 image to DCT array 
    '''
    DCT_64 = DCT(rgb_image_64).astype(np.uint8)
    '''
    Dividing this image into 4 different quaters of size 32*32
    '''
    x0 = DCT_64[:32, :32]
    x1 = DCT_64[:32, 32:64]
    x2 = DCT_64[32:65, :32]
    x3 = DCT_64[32:65, 32:65]

    '''
    DCT of 32 * 32 image 
    '''
    DCT_32 = DCT(rgb_image_32).astype(np.uint8)
    return np.concatenate((x0, x1, x2, x3, DCT_32), axis=2).astype(np.uint8)


def dct2(pix):
    return dct(dct(pix, axis=0, norm='ortho'), axis=1, norm='ortho')


# inverse dct2
def idct2(pix):
    return idct(idct(pix, axis=0, norm='ortho'), axis=1, norm='ortho')


def DCT(image):
    dct_R = dct2(image[:, :, 0])
    dct_G = dct2(image[:, :, 1])
    dct_B = dct2(image[:, :, 2])
    return np.dstack([dct_R, dct_G, dct_B])


def IDCT(image):
    idct_R = idct2(image[:, :, 0])
    idct_G = idct2(image[:, :, 1])
    idct_B = idct2(image[:, :, 2])
    return np.dstack([idct_R, idct_G, idct_B])


def save_images(images):
    count = 1
    for image in images:
        data = Image.fromarray(image.astype(np.uint8))
        data.save(str(count) + ".png")
        count = count + 1


def plot_an_image(images):
    count = 1
    plt.figure()
    for image in images:
        plt.subplot(2, 5, count)
        plt.imshow(image)
        plt.axis('off')
        plt.title("Image  " + str(count))
        count = count + 1
    plt.show()


if __name__ == '__main__':
    # preprocess_demosaic("", True)
    image = Image.open('../sample_images/01/01/001.png')
    transfom_to_multiple_images(image)
