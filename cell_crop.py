import os
from os.path import join

import cv2

mask_dir = 'C:\\Users\\park\\Downloads\\U-Net_0_20211220_112414\\Label (Segmentation)'
image_dir = 'C:\\Users\\park\\Downloads\\U-Net_0_20211220_112414\\Image'
cell_cropping_dir = 'C:\\Users\\park\\Downloads\\U-Net_0_20211220_112414'

files = os.listdir(mask_dir)

opti_threshold = 0.5 * 255

for f_num, file_name in enumerate(files):
    image = cv2.imread(join(image_dir,file_name))
    mask = cv2.imread(join(mask_dir, file_name),0)
    mask = cv2.resize(mask, dsize=(image.shape[1], image.shape[0]))
    mask[mask<opti_threshold] = 0
    mask[mask>=opti_threshold] = 1
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        cropped_img = image[y: y + h, x: x + w, :]
        cv2.imwrite(join(cell_cropping_dir, file_name[:-4] + '_' + str(x) + '_' + str(y) + '_' + str(w) + '_' + str(h) + '.png'), cropped_img)
