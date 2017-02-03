import cv2
import numpy as np


# cs is short for color space
def cvt_color(img, cs):
    color = None
    if cs[0] == 'gray':
        color = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif cs[0] == 'HLS':
        color = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        if cs[1] == 'H':
            color = color[:, :, 0]
        elif cs[1] == 'L':
            color = color[:, :, 1]
        else:
            color = color[:, :, 2]
    return color


def abs_sobel_thresh(img, orient='x', cs=('gray', None), sobel_kernel=3, thresh=(0, 255)):
    color = cvt_color(img, cs)
    dx = (orient == 'x')
    dy = (orient == 'y')
    sobel = cv2.Sobel(color, cv2.CV_64F, dx, dy, ksize=sobel_kernel)
    abs_sobel = np.abs(sobel)
    scaled_sobel = np.uint8(255 * (abs_sobel / np.max(abs_sobel)))

    grad_binary = np.zeros_like(scaled_sobel)
    grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return grad_binary


def mag_thresh(img, cs=('gray', None), sobel_kernel=3, mag_thresh=(0, 255)):
    color = cvt_color(img, cs)
    sobelx = cv2.Sobel(color, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(color, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    mag = np.sqrt(sobelx ** 2 + sobely ** 2)
    mag = np.uint8(255 * (mag / np.max(mag)))
    mag_binary = np.zeros_like(mag)
    mag_binary[(mag >= mag_thresh[0]) & (mag <= mag_thresh[1])] = 1
    return mag_binary


def dir_threshold(img, cs=('gray', None), sobel_kernel=3, thresh=(0, np.pi / 2)):
    color = cvt_color(img, cs)
    sobelx = cv2.Sobel(color, cv2.CV_64F, 1, 0, sobel_kernel)
    sobely = cv2.Sobel(color, cv2.CV_64F, 0, 1, sobel_kernel)

    abs_sobelx = np.abs(sobelx)
    abs_sobely = np.abs(sobely)

    dir_gradient = np.arctan2(abs_sobely, abs_sobelx)

    mask = np.zeros_like(dir_gradient)
    mask[(dir_gradient > thresh[0]) & (dir_gradient < thresh[1])] = 1

    return mask


def get_threshold_img(undistort, ksize):
    # gradient and color space
    cs_s = ('HLS', 'S')
    gradx_s = abs_sobel_thresh(undistort, orient='x', cs=cs_s, sobel_kernel=ksize, thresh=(20, 100))
    grady_s = abs_sobel_thresh(undistort, orient='y', cs=cs_s, sobel_kernel=ksize, thresh=(20, 100))
    mag_binary_s = mag_thresh(undistort, cs=cs_s, sobel_kernel=ksize, mag_thresh=(30, 100))
    dir_binary_s = dir_threshold(undistort, cs=cs_s, sobel_kernel=ksize, thresh=(0.7, 1.3))

    cs_gray = ('gray', None)
    gradx_gray = abs_sobel_thresh(undistort, orient='x', cs=cs_gray, sobel_kernel=ksize, thresh=(20, 100))
    grady_gray = abs_sobel_thresh(undistort, orient='y', cs=cs_gray, sobel_kernel=ksize, thresh=(20, 100))
    mag_binary_gray = mag_thresh(undistort, cs=cs_gray, sobel_kernel=ksize, mag_thresh=(30, 100))
    dir_binary_gray = dir_threshold(undistort, cs=cs_gray, sobel_kernel=ksize, thresh=(0.7, 1.3))

    threshold_img = np.zeros_like(dir_binary_gray)
    threshold_img[((gradx_s == 1) & (grady_s == 1)) | ((mag_binary_s == 1) & (dir_binary_s == 1)) | \
                  ((gradx_gray == 1) & (grady_gray == 1)) | ((mag_binary_gray == 1) & (dir_binary_gray == 1))] = 1
    return threshold_img
