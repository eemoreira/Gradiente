import numpy as np
import matplotlib.pyplot as plt

def gaussian_kernel(size, sigma=1):
    kernel = np.fromfunction(
        lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x - (size-1)/2)**2 + (y - (size-1)/2)**2) / (2*sigma**2)),
        (size, size)
    )
    return kernel / np.sum(kernel)

def apply_gaussian_filter(image, kernel_size=5, sigma=1):
    kernel = gaussian_kernel(kernel_size, sigma)
    filtered_image = convolution(image, kernel)
    return filtered_image

def convolution(image, kernel):
    kernel_size = kernel.shape[0]
    pad_width = kernel_size // 2
    padded_image = np.pad(image, pad_width, mode='reflect')
    filtered_image = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded_image[i:i+kernel_size, j:j+kernel_size]
            filtered_image[i, j] = np.sum(region * kernel)
    return filtered_image


def filter_image(image):
    to = plt.imread(image)
    if to.ndim == 3:
        to = np.mean(to, axis=2)
    filtered_image = apply_gaussian_filter(to, kernel_size=5, sigma=1)
    return filtered_image

def sobel(image):
    matx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    maty = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    return convolution(image, matx), convolution(image, maty)

def prewitt(image):
    matx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    maty = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    return convolution(image, matx), convolution(image, maty)

def scharr(image):
    matx = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]])
    maty = np.array([[3, 10, 3], [0, 0, 0], [-3, -10, -3]])
    return convolution(image, matx), convolution(image, maty)

def gradient_magnitude(Gx, Gy):
    n = Gx.shape[0]
    m = Gx.shape[1]
    ans = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            ans[i, j] = np.sqrt(Gy[i, j] * Gy[i, j] + Gx[i, j] * Gx[i, j])
    return ans

EPS = 1e-8

def gradient_direction(Gx, Gy):
    n = Gx.shape[0]
    m = Gx.shape[1]
    ans = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            ans[i, j] = np.arctan2(Gy[i, j], (Gx[i, j] + EPS))
    return ans

def contour(K, direction, magnitude):
    n, m = direction.shape
    PI = np.pi
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

            if mag >= K * max(neighbors): #>= or > ((min or max))??
                mask[i, j] = True
    return mask

def apply_contour(filtered_image, contorno, name):
    output_image = np.zeros_like(filtered_image, dtype=int)
    output_image = np.where(contorno == True, 255, 0)

    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(filtered_image, cmap='gray')

    plt.subplot(1, 2, 2)
    plt.title("Contorno")
    plt.imshow(output_image, cmap='gray')

    plt.savefig(f'{name}_From_Scratch.png')

images = ['insetoGray.png', 'moedas.png', 'Lua1_gray.jpg',
          'chessboard_inv.png', 'img02.jpg']
K = 1.1

for image_name in images:
    filtered_image = filter_image(f'img/{image_name}')
    filtered_image = filtered_image.astype(float)
    for filter_name, gradient_func in zip(["Prewitt", "Sobel", "Scharr"], [prewitt, sobel, scharr]):
        Gx, Gy = gradient_func(filtered_image)
        magnitude = gradient_magnitude(Gx, Gy)
        direction = gradient_direction(Gx, Gy)
        contour_mask = contour(K, direction, magnitude)
        apply_contour(filtered_image, contour_mask, f"{image_name[:-4]}_{filter_name}")

