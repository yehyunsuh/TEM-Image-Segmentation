import cv2
import numpy as np

# function for geting border and giving weights
def get_border(args, img):
    img_size = args.image_resize
    weights = args.weight
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)) # kernel for dilate & erode
    
    resized_img = cv2.resize(img, dsize=(img_size,img_size))
    
    # get eroded & dilated images
    ero_img = cv2.erode(resized_img, k)
    dil_img = cv2.dilate(resized_img, k)
    
    if args.s_weight:
        b_pixel = 0
        for i in range(img_size):
            for j in range(img_size):
                if (ero_img[i][j] != dil_img[i][j]).all():
                    b_pixel += 1
        weights = max(int((img_size*img_size-b_pixel)/b_pixel), 1)
    
    # get filters
    w_mask = np.ones((img_size, img_size))
    for i in range(img_size):
        for j in range(img_size):
            if (ero_img[i][j] != dil_img[i][j]).all():
                w_mask[i][j] = weights
    
    return w_mask