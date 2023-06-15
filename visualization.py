"""
reference
    normalize image:
        https://stackoverflow.com/questions/1735025/how-to-normalize-a-numpy-array-to-within-a-certain-range
    OSError: cannot write mode F as PNG:
        https://stackoverflow.com/questions/16720682/pil-cannot-write-mode-f-to-jpeg
    TypeError: Cannot handle this data type: (1, 1, 512), <f4:
        This happens because of Tensor has values between [0,1] and data type does not match
        https://stackoverflow.com/questions/60138697/typeerror-cannot-handle-this-data-type-1-1-3-f4
    KeyError: ((1, 1, 512), '|u1'):
        This happens because the input for Image.fromarray() should be (height,width) for grayscale images
        Therefore change (1,1,512,512) -> (512,512)
        https://stackoverflow.com/questions/57621092/keyerror-1-1-1280-u1-while-using-pils-image-fromarray-pil
"""

import os
import numpy as np
import torchvision

from PIL import Image


def create_directories(args):
    if not os.path.exists(f'./visualization'):
        os.mkdir(f'./visualization')
    if not os.path.exists(f'./visualization/{args.wandb_name}'):
        os.mkdir(f'./visualization/{args.wandb_name}')
    if not os.path.exists(f'./visualization/{args.wandb_name}/original'):
        os.mkdir(f'./visualization/{args.wandb_name}/original')
    if not os.path.exists(f'./visualization/{args.wandb_name}/prediction'):
        os.mkdir(f'./visualization/{args.wandb_name}/prediction')
    if not os.path.exists(f'./visualization/{args.wandb_name}/label'):
        os.mkdir(f'./visualization/{args.wandb_name}/label')


def original_image(args, image, idx):
    image_normalized = image * (1.0/image.max())

    for i in range(len(image_normalized[0])):
        torchvision.utils.save_image(image_normalized[0][i], f'./visualization/{args.wandb_name}/original/image_{idx}.png')


def original_image_with_label(args, image, label, idx):
    image_normalized = image * (1.0/image.max())
    for i in range(len(image_normalized[0])):
        torchvision.utils.save_image(image_normalized[0][i], f'./visualization/{args.wandb_name}/label/label_{idx}_image.png')
        torchvision.utils.save_image(label[0][i], f'./visualization/{args.wandb_name}/label/label_{idx}_label.png')

    ## Code for original image + label
    # image_normalized_255 = image * (255.0/image.max())
    # image_normalized_255 = image_normalized_255.detach().cpu().numpy()
    # image_normalized_255 = image_normalized_255.astype(np.uint8)

    # # label = label * 255
    # label = label.detach().cpu().numpy()
    # label = label.astype(np.uint8)

    # for i in range(len(image[0])):
    #     overlaid_image = Image.blend(Image.fromarray(image_normalized_255[i][0]), Image.fromarray(label[i][0]), 1)
    #     overlaid_image.save(f'./visualization/{args.wandb_name}/label/label_{idx}.png')


def original_image_with_prediction(args, image, predictions_binary, idx, epoch):
    image_normalized = image * (255.0/image.max())
    image_normalized = image_normalized.detach().cpu().numpy()
    image_normalized = image_normalized.astype(np.uint8)
    
    predictions_binary = predictions_binary.detach().cpu().numpy()
    predictions_binary = predictions_binary * 255
    predictions_binary = predictions_binary.astype(np.uint8)
    
    for i in range(len(image[0])):
        overlaid_image = Image.blend(Image.fromarray(image_normalized[i][0]), Image.fromarray(predictions_binary[i][0]), 0.2)
        overlaid_image.save(f'./visualization/{args.wandb_name}/prediction/epoch_{epoch}_prediction_{idx}.png')