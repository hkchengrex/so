import os
from os import path
import cv2
import numpy as np

from argparse import ArgumentParser
from multiprocessing import Pool


def create_overlay(image, mask):
    """
    image: H*W*3 numpy array
    mask: H*W numpy array
    If dimensions do not match, the mask is upsampled to match that of the image

    Returns a H*W*3 numpy array
    """
    h, w = image.shape[:2]
    mask = cv2.resize(mask, dsize=(w,h), interpolation=cv2.INTER_CUBIC)

    # color options: https://docs.opencv.org/4.x/d3/d50/group__imgproc__colormap.html
    mask_color = cv2.applyColorMap(mask, cv2.COLORMAP_HOT).astype(np.float32)
    mask = mask[:, :, None] # create trailing dimension for broadcasting
    mask = mask.astype(np.float32)/255

    # different other options that you can use to merge image/mask
    overlay = (image*(1-mask)+mask_color*mask).astype(np.uint8)
    # overlay = (image*0.5 + mask_color*0.5).astype(np.uint8)
    # overlay = (image + mask_color).clip(0,255).astype(np.uint8)

    return overlay

def process_video(video_name):
    """
    Processing frames in a single video
    """
    vid_image_path = path.join(image_path, video_name)
    vid_mask_path = path.join(mask_path, video_name)
    vid_output_path = path.join(output_path, video_name)
    os.makedirs(vid_output_path, exist_ok=True)

    frames = sorted(os.listdir(vid_image_path))
    for f in frames:
        image = cv2.imread(path.join(vid_image_path, f))
        mask = cv2.imread(path.join(vid_mask_path, f.replace('.jpg','.png')), cv2.IMREAD_GRAYSCALE)
        overlay = create_overlay(image, mask)
        cv2.imwrite(path.join(vid_output_path, f), overlay)


parser = ArgumentParser()
parser.add_argument('--image_path')
parser.add_argument('--mask_path')
parser.add_argument('--output_path')
args = parser.parse_args()

image_path = args.image_path
mask_path = args.mask_path
output_path = args.output_path

if __name__ == '__main__':
    videos = sorted(
        list(set(os.listdir(image_path)).intersection(
                set(os.listdir(mask_path))))
    )

    print(f'Processing {len(videos)} videos.')

    pool = Pool()
    pool.map(process_video, videos)

    print('Done')
