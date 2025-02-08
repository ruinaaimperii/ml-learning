import cv2
import numpy as np
import scipy
import matplotlib.pyplot as plt


i = scipy.datasets.ascent()
plt.gray()
plt.imshow(i)
plt.show()

conv_matrix = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
weight = 1


# applying conv_matrix
def convolution_filter(gray_img, kernel):
    # temporary operation
    # gray_img = cv2.resize(gray_img, (10, 10))

    kernel_size = len(kernel)

    row = gray_img.shape[0] - kernel_size + 1
    col = gray_img.shape[1] - kernel_size + 1

    result = np.zeros(shape=(row, col))

    for i in range(row):
        for j in range(col):
            current = gray_img[i:i+kernel_size, j:j+kernel_size]
            multiplication = np.abs(sum(sum(current * kernel)))
            result[i, j] = multiplication

    return result


# i_transformed = convolution_filter(i, conv_matrix)
def apply_3x3_convolution(i:np.ndarray, conv_matrix:np.ndarray or list, weight:float) -> np.ndarray:
    size_x = i.shape[0]
    size_y = i.shape[1]
    result = np.zeros(shape=(size_x, size_y))
    for x in range(1, size_x - 1):
        for y in range(1, size_y - 1):
            output_pixel = 0.0
            output_pixel = output_pixel + (i[x - 1, y-1] * conv_matrix[0][0])
            output_pixel = output_pixel + (i[x, y - 1] * conv_matrix[0][1])
            output_pixel = output_pixel + (i[x + 1, y - 1] * conv_matrix[0][2])
            output_pixel = output_pixel + (i[x - 1, y] * conv_matrix[1][0])
            output_pixel = output_pixel + (i[x, y] * conv_matrix[1][1])
            output_pixel = output_pixel + (i[x + 1, y] * conv_matrix[1][2])
            output_pixel = output_pixel + (i[x - 1, y + 1] * conv_matrix[2][0])
            output_pixel = output_pixel + (i[x, y + 1] * conv_matrix[2][1])
            output_pixel = output_pixel + (i[x + 1, y + 1] * conv_matrix[2][2])
            output_pixel *= weight
            if output_pixel < 0:
                output_pixel = 0
            if output_pixel > 255:
                output_pixel = 255
            result[x][y] = output_pixel
    return result


# i_transformed = cv2.GaussianBlur(i, (5, 5), 10)
i_convoluted = apply_3x3_convolution(i, conv_matrix, weight)


def apply_2x2_pooling(i:np.ndarray) -> np.ndarray:
    size_x, size_y = i.shape
    result = np.zeros(shape=(size_x // 2, size_y // 2))
    for x in range(0, size_x, 2):
        for y in range(0, size_y, 2):
            element00 = i[x, y]
            element01 = i[x, y + 1]
            element10 = i[x + 1, y]
            element11 = i[x + 1, y + 1]
            result[x // 2, y // 2] = max(element00, element01, element10, element11)
    return result


i_pooled = apply_2x2_pooling(i_convoluted)
plt.gray()
plt.imshow(i_pooled)
plt.show()