import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_gaussian_filter(image, kernel_size=5, sigma=1):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

def prewitt(image):
    kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32)
    kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32)
    Gx = cv2.filter2D(image, -1, kernelx)
    Gy = cv2.filter2D(image, -1, kernely)
    return Gx, Gy

def sobel(image):
    Gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    Gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    return Gx, Gy

def scharr(image):
    Gx = cv2.Scharr(image, cv2.CV_64F, 1, 0)
    Gy = cv2.Scharr(image, cv2.CV_64F, 0, 1)
    return Gx, Gy

def gradient_magnitude(Gx, Gy):
    return np.sqrt(Gx**2 + Gy**2)

def gradient_direction(Gx, Gy):
    return np.arctan2(Gy, Gx)

def contour(K, direction, magnitude):
    n, m = direction.shape
    mask = np.zeros_like(direction, dtype=bool)
    for i in range(1, n - 1):
        for j in range(1, m - 1):
            mag = magnitude[i, j]
            dir_angle = direction[i, j]

            if (-np.pi/8 <= dir_angle < np.pi/8) or (7*np.pi/8 <= dir_angle or dir_angle < -7*np.pi/8):
                neighbors = [magnitude[i-1, j], magnitude[i+1, j]]
            elif (np.pi/8 <= dir_angle < 3*np.pi/8) or (-7*np.pi/8 <= dir_angle < -5*np.pi/8):
                neighbors = [magnitude[i-1, j+1], magnitude[i+1, j-1]]
            elif (3*np.pi/8 <= dir_angle < 5*np.pi/8) or (-5*np.pi/8 <= dir_angle < -3*np.pi/8):
                neighbors = [magnitude[i, j-1], magnitude[i, j+1]]
            else:
                neighbors = [magnitude[i-1, j-1], magnitude[i+1, j+1]]

            if mag >= K * max(neighbors):
                mask[i, j] = True

    return mask

def apply_contour(filtered_image, contour_mask, name):
    output_image = np.where(contour_mask, 255, 0).astype(np.uint8)

    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(filtered_image, cmap='gray')

    plt.subplot(1, 2, 2)
    plt.title("Contorno")
    plt.imshow(output_image, cmap='gray')

    plt.savefig(f'{name}_with_Opencv.png')

images = ['insetoGray.png', 'moedas.png', 'Lua1_gray.jpg', 'chessboard_inv.png', 'img02.jpg']

K = 1.1

for image_name in images:
    image = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
    filtered_image = apply_gaussian_filter(image, kernel_size=5, sigma=1)

    for filter_name, gradient_func in zip(["Prewitt", "Sobel", "Scharr"], [prewitt, sobel, scharr]):
        Gx, Gy = gradient_func(filtered_image)
        magnitude = gradient_magnitude(Gx, Gy)
        direction = gradient_direction(Gx, Gy)
        contour_mask = contour(K, direction, magnitude)
        apply_contour(filtered_image, contour_mask, f"{image_name[:-4]}_{filter_name}")

