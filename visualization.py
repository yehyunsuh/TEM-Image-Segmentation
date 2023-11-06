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
    if not os.path.exists(f'./visualization/{args.wandb_name}/prediction_color'):
        os.mkdir(f'./visualization/{args.wandb_name}/prediction_color')
    

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
        
        
def original_image_with_prediction_color(args, image, predictions_binary, label, idx, epoch):
    image_normalized = image * (255.0/image.max())
    image_normalized = image_normalized.detach().cpu().numpy()
    image_normalized = image_normalized.astype(np.uint8)

    label_normalized = label*(1/label.max())
    label_normalized = label_normalized.detach().cpu().numpy()

    predictions_binary = predictions_binary.detach().cpu().numpy()

    for i in range(len(image)):
        p_b = predictions_binary[i][0]
        y_b = label_normalized[i][0]

        r = y_b - p_b # predict O but X
        r = r.clip(0)
        r = r.astype(np.uint8)

        b = p_b - y_b # predict X but O
        b = b.clip(0)
        b = b.astype(np.uint8)

        g = np.zeros(r.shape)

        mask = np.stack((r,g,b),axis=0)
        mask = np.transpose(mask, (1, 2, 0)) # [C,H,W] -> [H,W,C]
        mask = mask*255

        img = image_normalized[i]
        img = np.transpose(img, (1, 2, 0)) # [C,H,W] -> [H,W,C]

        blended = img*0.4 + mask*0.6
        blended = blended.astype(np.uint8)

        blended = Image.fromarray(blended)
        blended.save(f'./visualization/{args.wandb_name}/prediction_color/epoch_{epoch}_prediction_color_{idx}.png')
        
        return blended
        

def image_w_heatmap(args, idx, image_name, epoch, prediction):
    for i in range(len(prediction[0])):
        plt.imshow(prediction[0][i].detach().cpu().numpy(), interpolation='nearest')
        plt.axis('off')
        plt.savefig(f'./results/{args.wandb_name}/heatmap/label{i}/{image_name}_{epoch}_heatmap.png', bbox_inches='tight', pad_inches=0, dpi=150)