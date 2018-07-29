import cv2
import numpy as np

# Create two images
im1 = np.zeros((100, 100, 3), np.uint8)
im1[:] = (255, 0, 0)
im2 = np.zeros((100, 100, 3), np.uint8)
im2[:] = (0, 255, 0)

# Generate a random mask
ran = np.random.randint(0, 2, (100, 100), np.uint8)

# List of images and masks
images = [im1, im2]
mask = [ran, 1-ran]

not_output = np.zeros((100, 100, 3), np.uint8)
copy_output = np.zeros((100, 100, 3), np.uint8)

for i in range(0, len(images)):
    # Using the 'NOT' way
    not_output = cv2.bitwise_not(images[i], not_output, mask=mask[i])
    # Using the copyto way
    np.copyto(copy_output, images[i], where=mask[i][:, :, None].astype(bool))

cv2.imwrite('not.png', 255 - not_output)
cv2.imwrite('copy.png', copy_output)
