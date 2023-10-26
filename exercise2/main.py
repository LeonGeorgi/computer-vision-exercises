import cv2
import numpy as np

import util as util
from exercise2.util import plot_histogram_and_cumulative_distribution, calculate_histogram, \
    calculate_cumulative_distribution

KEYCODE_ESC = 27
KEYCODE_SPACE = 32

STATUS_CONTINUE = 0
STATUS_QUIT_APPLICATION = 1

show_histograms = False


def main():
    print("\nTU Dresden, Inf, CV1 Ex2, Holger Heidrich")
    print("Press Space to continue")
    print("Press Esc to quit the application")

    filename = "../assets/fruits_gray_low_contrast.jpg"
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

    cv2.imshow("Fruits!", image)

    if wait_for_and_process_key_input() == STATUS_QUIT_APPLICATION:
        exit()

    task_1_automatic_equilibration(image)
    task_2_implement_equilibration(image)

    cv2.destroyWindow("Fruits!")

    noisy_img = cv2.imread("../assets/fruits_noisy.jpg", cv2.IMREAD_GRAYSCALE)
    cv2.imshow("Noisy Fruits", noisy_img)

    if wait_for_and_process_key_input() == STATUS_QUIT_APPLICATION:
        exit()

    task_4_box_and_gaussian_filter(noisy_img)
    sobel_filter_output_horizontal, sobel_filter_output_vertical = task_5_sobel_filter(noisy_img)
    task_6_gradient_magnitude(noisy_img, sobel_filter_output_horizontal, sobel_filter_output_vertical)

    cv2.destroyAllWindows()

    if wait_for_and_process_key_input() == STATUS_QUIT_APPLICATION:
        exit()


def task_1_automatic_equilibration(image):
    equalized_image = cv2.equalizeHist(image)
    show_image("Equalized Fruits", equalized_image)


def task_2_implement_equilibration(image):
    histogram = calculate_histogram(image)
    cumulative_distribution = calculate_cumulative_distribution(histogram, image)
    if show_histograms:
        plot_histogram_and_cumulative_distribution(cumulative_distribution, histogram)

    equalized_img = np.zeros_like(image)
    height, width = image.shape
    for row in range(height):
        for col in range(width):
            equalized_img[row, col] = 255 * cumulative_distribution[image[row, col]]
    if show_histograms:
        histogram = calculate_histogram(equalized_img)
        cumulative_distribution = calculate_cumulative_distribution(histogram, equalized_img)
        plot_histogram_and_cumulative_distribution(cumulative_distribution, histogram)
    show_image("Equalized Fruits", equalized_img)


def task_4_box_and_gaussian_filter(noisy_img):
    box_filter_output = np.zeros_like(noisy_img)
    util.apply_filter(noisy_img, box_filter_output, util.box_filter)
    show_image("Box Filter Image", box_filter_output)
    # Task 4.2
    gauss_filter_output = np.zeros_like(noisy_img)
    util.apply_filter(noisy_img, gauss_filter_output, util.gauss_filter)
    show_image("Gauss Filter Image", gauss_filter_output)


def task_5_sobel_filter(noisy_img):
    # Task 5.1: Horizontal Sobel Filter
    sobel_filter_output_horizontal = np.zeros_like(noisy_img, dtype=np.int32)
    util.apply_filter(noisy_img, sobel_filter_output_horizontal, util.sobel_filter_horizontal, color_offset=0)
    show_image("Sobel Filter Image", (sobel_filter_output_horizontal + 128).astype(np.uint8))
    # Task 5.2: Vertical Sobel Filter
    sobel_filter_output_vertical = np.zeros_like(noisy_img, dtype=np.int32)
    util.apply_filter(noisy_img, sobel_filter_output_vertical, util.sobel_filter_vertical,
                      color_offset=0)
    show_image("Sobel Filter Image", (sobel_filter_output_vertical + 128).astype(np.uint8))
    return sobel_filter_output_horizontal, sobel_filter_output_vertical


def task_6_gradient_magnitude(noisy_img, sobel_filter_output_horizontal, sobel_filter_output_vertical):
    # Task 6.1: Gradient Magnitude: fruits_noisy.jpg
    gradient_magnitude = np.sqrt(np.square(sobel_filter_output_horizontal) + np.square(sobel_filter_output_vertical))
    gradient_magnitude = gradient_magnitude / gradient_magnitude.max() * 255
    gradient_magnitude = np.round(gradient_magnitude).astype(np.uint8)
    cv2.imshow("Gradient Magnitude fruits_noisy.jpg", gradient_magnitude)
    if wait_for_and_process_key_input() == STATUS_QUIT_APPLICATION:
        exit()

    # Task 6.2: Gradient Magnitude: fruits_noisy.jpg with Gaussian Filter
    gauss_filter_output = np.zeros_like(noisy_img)
    util.apply_filter(noisy_img, gauss_filter_output, util.gauss_filter)
    sobel_filter_output_horizontal = np.zeros_like(gauss_filter_output, dtype=np.int32)
    util.apply_filter(gauss_filter_output, sobel_filter_output_horizontal, util.sobel_filter_horizontal, color_offset=0)
    sobel_filter_output_vertical = np.zeros_like(gauss_filter_output, dtype=np.int32)
    util.apply_filter(gauss_filter_output, sobel_filter_output_vertical, util.sobel_filter_vertical, color_offset=0)
    gradient_magnitude = np.sqrt(np.square(sobel_filter_output_horizontal) + np.square(sobel_filter_output_vertical))
    gradient_magnitude = gradient_magnitude / gradient_magnitude.max() * 255
    gradient_magnitude = np.round(gradient_magnitude).astype(np.uint8)
    show_image("Gradient Magnitude fruits_noisy.jpg with Gaussian Filter", gradient_magnitude)


def wait_for_and_process_key_input():
    while True:
        keycode = cv2.waitKey(0) & 0xEFFFFF

        if keycode == KEYCODE_ESC:
            return STATUS_QUIT_APPLICATION
        elif keycode == KEYCODE_SPACE:
            return STATUS_CONTINUE


def show_image(window_name, image):
    cv2.imshow(window_name, image)
    if wait_for_and_process_key_input() == STATUS_QUIT_APPLICATION:
        exit()
    cv2.destroyWindow(window_name)


if __name__ == "__main__":
    main()
