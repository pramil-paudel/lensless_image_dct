import os
from matplotlib import image as mpimg
from scipy.io import loadmat
import numpy as np
import cv2
from scipy.fftpack import dct, idct
import flatcam
import imageio
from matplotlib import pyplot as plt
from PIL import Image
from skimage.transform import resize

# calib = loadmat('flatcam_calibdata.mat')  # load calibration data


# Reconstruct
lmbd = 3e-4  # L2 regularization parameter


def preprocess_demosaic(data_path, test):
    '''
    :param data_path: folder of lensless images from rice university
    :return: Bayers Pattern Images
    '''

    global calib
    root_dir = [x for x in os.walk(os.path.join(data_path, 'sample_images'))]

    '''
    creating new data path for demosaic images 
    '''

    root_new = os.path.join(data_path, 'demosaiced_measurement')

    for sub_dir in root_dir[0][1]:
        demosaic_dir = os.path.join(root_new, sub_dir)
        if not os.path.isdir(demosaic_dir):
            os.makedirs(demosaic_dir)
    for sub_dir in root_dir[0][1]:
        demosaic_dir = os.path.join(root_new, sub_dir)
        if not os.path.isdir(demosaic_dir):
            os.makedirs(demosaic_dir)

    for c, sub_dir in enumerate(root_dir[1:]):
        class_dir = sub_dir[0]
        image_list = sub_dir[2]
        for i, image_name in enumerate(image_list):
            # Calib Data
            calib = loadmat('flatcam_calibdata.mat')
            flatcam.clean_calib(calib)
            # Input Image
            input_image = mpimg.imread(os.path.join(class_dir, image_name))
            '''
            Covert into Bayers Pattern 
            '''
            bayers_pattern = flatcam.fc2bayer(input_image, calib)
            '''
            Coverting into RGB Pattern (lets try without normalization ) 
            '''
            rgb_image = flatcam.bayer2rgb(bayers_pattern)
            rgb_image_in_255 = rgb_image*255
            '''
            First resize into (64,64) and take DCT : it will convert into 
            '''
            rgb_image_64 = resize(rgb_image_in_255, (64, 64))
            rgb_image_32 = resize(rgb_image_in_255, (32, 32))
            # This resize has different output ???
            # rgb_image_64 = cv2.resize(rgb_image, (64, 64), interpolation=cv2.INTER_CUBIC)
            # rgb_image_32 = cv2.resize(rgb_image, (32, 32), interpolation=cv2.INTER_CUBIC)
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

            # plotting all images in the form :
            # Saving all images for testing
            if test :
                plot_an_image([DCT_64, x0, x1, x2, x3, DCT_32])
                save_images([DCT_64, x0, x1, x2, x3, DCT_32])
            merged_array = np.concatenate((x0, x1, x2, x3, DCT_32), axis=2).astype(np.uint8)
            #image_to_feed = Image.fromarray(merged_array.astype(np.uint8))


            # demosaiced_img = cv2.resize(demosaiced_img, (64, 64), interpolation=cv2.INTER_CUBIC)
            class_name = os.path.split(class_dir)[1]
            imageio.imwrite(os.path.join(root_new, class_name, image_name), merged_array)
            # imageio.imwrite(os.path.join(root_new, class_name, image_name), rgb_image_32.astype(np.uint8))
            print(imageio.imread(os.path.join(root_new, class_name, image_name)).shape)
        print(f"Conversion done for {class_dir.split('/')[-1]}")


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
        data  = Image.fromarray(image.astype(np.uint8))
        data.save(str(count)+".png")
        count = count+1


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
    preprocess_demosaic("", True)
