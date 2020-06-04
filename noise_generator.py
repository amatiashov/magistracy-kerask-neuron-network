import os
import uuid

import cv2
import random
import numpy as np
from constants import BATHES_DIR


def gauss_noisy(image):
    row, col, ch = image.shape
    mean = 0
    var = 0.1
    sigma = var**0.9
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    return image + gauss


def poisson_noisy(image):
    vals = len(np.unique(image))
    vals = 2 ** np.ceil(np.log2(vals))
    noisy = np.random.poisson(image * vals) / float(vals)
    return noisy


def speckle_noisy(image):
    row, col, ch = image.shape
    gauss = np.random.randn(row, col, ch)
    gauss = gauss.reshape(row, col, ch)
    return image + image * gauss


def sp_noise(image, prob):
    """
    Add salt and pepper noise to image
    :param image: image object
    :param prob: Probability of the noise
    :return: noisy image
    """
    output = np.zeros(image.shape, np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output


source_img_obj = cv2.imread('source.png', 1)


# generate train objects
noise_img = speckle_noisy(source_img_obj)
cv2.imwrite(os.path.join(BATHES_DIR, "train", "true", "speckle_noisy-%s.png" % str(uuid.uuid4())), noise_img)

noise_img = sp_noise(source_img_obj, 0.1)
cv2.imwrite(os.path.join(BATHES_DIR, "train", "true", "sp_noisy-%s.png" % str(uuid.uuid4())), noise_img)

noise_img = poisson_noisy(source_img_obj)
cv2.imwrite(os.path.join(BATHES_DIR, "train", "true", "poisson_noisy-%s.png" % str(uuid.uuid4())), noise_img)


# generate test objects
noise_img = speckle_noisy(source_img_obj)
cv2.imwrite(os.path.join(BATHES_DIR, "test", "true", "speckle_noisy-%s.png" % str(uuid.uuid4())), noise_img)

noise_img = sp_noise(source_img_obj, 0.1)
cv2.imwrite(os.path.join(BATHES_DIR, "test", "true", "sp_noisy-%s.png" % str(uuid.uuid4())), noise_img)

noise_img = poisson_noisy(source_img_obj)
cv2.imwrite(os.path.join(BATHES_DIR, "test", "true", "poisson_noisy-%s.png" % str(uuid.uuid4())), noise_img)
