import numpy as np
import cv2

image = cv2.imread("arr.png", cv2.IMREAD_UNCHANGED)
b, g, r, a = cv2.split(image)

bgr = cv2.merge((b, g, r))

all_g = np.ones_like(bgr)
all_g[:, :] = (0,255,0)

bgr = np.where(bgr == (0,0,255), all_g, bgr)

image = cv2.merge((bgr, a))

cv2.imwrite('target.png', image)
