import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as F
import pandas as pd
import os
import PIL

plt.rcParams["savefig.bbox"] = 'tight'

def show(imgs, return_grid=False, figsize=(12,8), ncols = None):
    if not isinstance(imgs, list):
        imgs = [imgs]
    if ncols is None:
        ncols = len(imgs)
    fix, axs = plt.subplots(figsize=figsize, ncols=ncols, squeeze=False)
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
    
    
def plot_train_val_loss(train_loss = None, val_loss = None, **kwargs):
    """plot train and validation loss
    Args:
        train_loss (list or array): list of loss values
        val_loss (list or array): list of loss values
    """

    fig = plt.subplots(figsize=(14, 4))
    if train_loss:
        if val_loss:
            plt.subplot(1,2,1)
        p = sns.lineplot(x=np.arange(len(train_loss)), 
                         y=train_loss, label='Loss')
        p.set_title('Training Loss', loc='left')
        p.set_xlabel('Batches')
        p.set_ylabel('Loss')
        sns.despine()
    if val_loss:
        if train_loss:
            plt.subplot(1,2,2)
        p = sns.lineplot(x=np.arange(len(val_loss)), y=val_loss,
                         label='Loss')
        p.set_title('Validation Loss', loc='left')
        p.set_xlabel('Batches')
        p.set_ylabel('Loss')
        sns.despine()

    plt.show()

    
def loss_sma(loss, sma=50, show_fig = True):
        """plot loss with moving average
        
        Args:
            loss (_type_): _description_
            sma (int, optional): _description_. Defaults to 50.
            show_fig (bool, optional): _description_. Defaults to True.
        """
        colors = sns.color_palette('Paired', 4)
        sns.set_style('white')
        if not isinstance(loss, pd.Series):
            loss = pd.Series(loss)

        mean_loss_folds = loss.rolling(sma).mean()
        std_loss_folds = loss.rolling(sma).std()

        p = sns.lineplot(x=mean_loss_folds.index, y=mean_loss_folds, label='Mean Batch', color=colors[1])
        p = sns.lineplot(x=mean_loss_folds.index, y=mean_loss_folds + std_loss_folds, 
                         label=r'$\pm1\sigma$', color=colors[0], linestyle='--', alpha=.5)
        p = sns.lineplot(x=mean_loss_folds.index, y=mean_loss_folds - std_loss_folds, 
                         color=colors[0], linestyle='--', alpha=.5)
        plt.text(x=mean_loss_folds.index[-1], y=mean_loss_folds.iloc[-1], 
                 s=str(round(mean_loss_folds.iloc[-1], 2)), va='center')

        p.set_title(f'Loss over Batches / SMA{50}',loc='left')
        p.set_xlabel('Batches')
        p.set_ylabel('Loss')
        sns.despine()
        if show_fig:
            plt.show()
            

def visualize_overlaps(n: int, y_true: list, y_pred: list, images: list, 
                       inverse_image_transforms = None):
    """
    Visualizes overlpas between image and segmentation mask.
    """
    mask_images_idx = []
    i = 0
    while len(mask_images_idx) < n:
        if y_true[i].sum() > 0:
            mask_images_idx.append(i)
        i += 1
    
    for idx in mask_images_idx:
        PLOT_ID = idx
        if inverse_image_transforms:
            true_mask = inverse_image_transforms(y_true[PLOT_ID])
            pred_mask = inverse_image_transforms(y_pred[PLOT_ID])
            image = inverse_image_transforms(images[PLOT_ID])

        fig, ax = plt.subplots(figsize=(12, 4))

        plt.subplot(1,3,1)
        plt.imshow(true_mask, 
                   cmap='gray', 
                   label='True', 
                   interpolation='none') 
        plt.axis('off')
        plt.title(f'True Mask {PLOT_ID}')

        plt.subplot(1,3,2)
        plt.imshow(pred_mask, 
                   cmap='jet', 
                   alpha=0.5, 
                   label='Predicted', 
                   interpolation='none') 
        plt.axis('off')
        plt.title(f'Predicted Mask {PLOT_ID}')

        plt.subplot(1,3,3)
        plt.imshow(image, cmap='gray')
        plt.imshow(true_mask, cmap='gray', 
                   label='True', 
                   interpolation='none', alpha=.3) 
        plt.imshow(pred_mask, 
                   cmap='jet', 
                   alpha=0.3, 
                   label='Predicted', 
                   interpolation='none') 
        plt.axis('off')
        plt.title('Overlay')

        plt.show()
