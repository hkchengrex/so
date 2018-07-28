import cv2
import matplotlib.pyplot as plt

IMG_SIZE = 224

im = cv2.cvtColor(cv2.imread('lena.jpg'), cv2.COLOR_BGR2GRAY)
im = cv2.resize(im, (IMG_SIZE, IMG_SIZE))

# Your detector results
detected_region = [
    [(10, 20)   , (80, 100)],
    [(50, 0)    , (220, 190)],
    [(100, 143)  , (180, 200)],
    [(110, 45)  , (180, 150)]
]

# Global states
x_scale = 1.0
y_scale = 1.0
x_shift = 0
y_shift = 0

x1, y1 = 0, 0
x2, y2 = IMG_SIZE-1, IMG_SIZE-1
i = 0
for region in detected_region:
    i += 1
    # Detection
    x_scale = IMG_SIZE / (x2-x1)
    y_scale = IMG_SIZE / (y2-y1)
    x_shift = x1
    y_shift = y1

    cur_im = cv2.resize(im[y1:y2, x1:x2], (IMG_SIZE, IMG_SIZE))

    # Assuming the detector return these results
    cv2.rectangle(cur_im, region[0], region[1], (255))

    plt.imshow(cur_im)
    plt.savefig('%d.png'%i, dpi=200)
    plt.show()

    # Zooming in, using part of your code
    context_pixels = 16
    x1 = max(region[0][0] - context_pixels, 0) / x_scale + x_shift
    y1 = max(region[0][1] - context_pixels, 0) / y_scale + y_shift
    x2 = min(region[1][0] + context_pixels, IMG_SIZE) / x_scale + x_shift
    y2 = min(region[1][1] + context_pixels, IMG_SIZE) / y_scale + y_shift

    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)


# Assuming the detector confirm its choice here
print('Confirmed detection: ', x1, y1, x2, y2)

# This time no padding
x1 = detected_region[-1][0][0] / x_scale + x_shift
y1 = detected_region[-1][0][1] / y_scale + y_shift
x2 = detected_region[-1][1][0] / x_scale + x_shift
y2 = detected_region[-1][1][1] / y_scale + y_shift
x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

cv2.rectangle(im, (x1, y1), (x2, y2), (255, 0, 0))
plt.imshow(im)
plt.savefig('final.png', dpi=300)
plt.show()