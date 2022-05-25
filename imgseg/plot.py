import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as F
import pandas as pd
import os
import PIL

plt.rcParams["savefig.bbox"] = 'tight'


def show(imgs, return_grid=False):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(figsize=(12,8), ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])   
    if return_grid:
        return fix
    else:
        plt.show()
        
        
def plot_random_masked_images(df: pd.DataFrame, n_images: int, 
                      image_path: str = './kaggle_3m/images/', 
                      mask_path: str = './kaggle_3m/masks/'):
    samples = df.loc[:, ['image', 'mask']].sample(n_images)
    ncol = 5
    nrows = int(np.floor(n_images / ncol))

    fig, ax = plt.subplots(figsize=(18, nrows*4))

    for i, (img, mask) in enumerate(samples.to_numpy(), 1):
        img = os.path.join(image_path, img)
        mask = os.path.join(mask_path, mask)
        plt.subplot(nrows, ncol, i)
        plt.imshow(PIL.Image.open(img), cmap='Greys')
        plt.imshow(PIL.Image.open(mask), alpha=.5)
        plt.title(f'{os.path.split(img)[-1]}', fontsize=8)
        plt.axis('off')
        
    plt.show()
    
def plot_one_image_mask(df: pd.DataFrame, img_name: str, path_masks: str = './kaggle_3m/masks/', 
                        path_images: str = './kaggle_3m/images/'):
    img_mask = df.loc[df['image'] == img_name, 'mask'].values[0]
    fig, ax = plt.subplots(figsize=(10, 4))
    plt.subplot(1,3,1)
    plt.imshow(PIL.Image.open(os.path.join(path_images, img_name)))
    plt.title('Image')
    plt.axis('off')
    plt.subplot(1,3,2)
    plt.imshow(PIL.Image.open(os.path.join(path_masks, img_mask)))
    plt.title('Mask')
    plt.axis('off')
    plt.subplot(1,3,3)
    plt.imshow(PIL.Image.open(os.path.join(path_images, img_name)))
    plt.imshow(PIL.Image.open(os.path.join(path_masks, img_mask)), alpha=.3)
    plt.title('Image + Mask')
    plt.axis('off')
    
    plt.show()