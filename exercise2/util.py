from collections import Counter

import cv2
import numpy as np
from matplotlib import pyplot as plt

box_filter = np.array([[1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1]]) / 9

gauss_filter = np.array([[1, 2, 1],
                         [2, 4, 2],
                         [1, 2, 1]]) / 16

sobel_filter_horizontal = np.array([[-1, 0, 1],
                                    [-2, 0, 2],
                                    [-1, 0, 1]])

sobel_filter_vertical = np.array([[-1, -2, -1],
                                  [0, 0, 0],
                                  [1, 2, 1]])

def apply_filter(input_img: np.ndarray, output_img, image_filter: np.ndarray, color_offset: int = 0):
    filter_height, filter_width = image_filter.shape
    destination_image_with_border = cv2.copyMakeBorder(input_img, filter_height // 2, filter_height // 2,
                                                       filter_width // 2, filter_width // 2,
                                                       cv2.BORDER_REFLECT)
    # Task 3
    for row in range(input_img.shape[0]):
        for col in range(input_img.shape[1]):
            row_start = row
            row_end = row_start + filter_height
            col_start = col
            col_end = col_start + filter_width
            value = (destination_image_with_border[row_start:row_end,
                     col_start:col_end] * image_filter).sum() + color_offset
            output_img[row, col] = value
            if value != output_img[row, col]:
                print("Overflow")


def plot_histogram_and_cumulative_distribution(cumulative_histogram, histogram):
    plt.figure(1)
    plt.subplot(211)
    plt.xlim([min(0, min(histogram.keys())), max(255, max(histogram.keys()))])
    plt.bar(list(histogram.keys()), list(histogram.values()))
    plt.subplot(212)
    plt.xlim([min(0, min(histogram.keys())), max(255, max(histogram.keys()))])
    plt.bar(list(cumulative_histogram.keys()), list(cumulative_histogram.values()))
    plt.show()


def calculate_histogram(image):
    height, width = image.shape
    histogram = Counter(image[row, col] for row in range(height) for col in range(width))
    return histogram


def calculate_cumulative_distribution(histogram, image):
    cumulative_sum = dict()
    height, width = image.shape
    for i in range(256):
        if i not in cumulative_sum:
            cumulative_sum[i] = 0
        if i == 0:
            cumulative_sum[i] = histogram[i] / (height * width)
        else:
            cumulative_sum[i] = cumulative_sum[i - 1] + (histogram[i] / (height * width))
    return cumulative_sum
