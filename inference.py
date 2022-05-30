import os
import argparse
from tqdm import tqdm
import PIL

import torch
import torchvision.transforms as transforms

from imgseg import UNet3, DoubleConvLayer

# add argparser
parser = argparse.ArgumentParser()
parser.add_argument(
    '--input', 
    '-i',
    help='input folder',
    default='./image_in/'
)
parser.add_argument(
    '--output',
    '-o',
    help='output folder',
    default='./image_out/'
)


def check_dirs(in_dir: str, out_dir: str):
    if not os.path.isdir(in_dir) and not os.path.isfile(in_dir):
        raise Exception('There is a problem with given input or output dir')
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
        
def load_images(img_dir: str):
    all_images = []
    for file in os.listdir(img_dir):
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
            img_file = os.path.join(img_dir, file)
            all_images.append(PIL.Image.open(img_file))
    
    return all_images

def inference(all_images):
    # transform all inputs
    model = torch.load('./models/segmodel_60eps.pth', map_location = 'cpu')
    
    # Define image transforms
    image_transforms_inference = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(size=(256, 256)),
        transforms.ToTensor()
    ]) 
    inverse_image_transforms = transforms.Compose([
        transforms.ToPILImage()
    ])
    
    # Transform image
    all_images = [image_transforms_inference(img) for img in all_images]

    # get model prediction
    #assert isinstance(all_images[0], torch.Tensor), 'Images are not type of tensor'
    segmentations = []
    with torch.no_grad():
        for image in all_images:
            predicted_mask = model(image.unsqueeze(0))
            predicted_mask = torch.where(predicted_mask >= .5, 1., 0.)
            segmentations.append(predicted_mask)
        
    return [inverse_image_transforms(img.squeeze(0)) for img in segmentations]

def save_model_output(image_names, output_dir, masks):
    new_image_names = ['masked_'+name for name in image_names]
    new_save_path = [os.path.join(output_dir, name) for name in new_image_names]
    for path, mask in zip(new_save_path, masks):
        mask.save(path)
        
    
def main(args):
    pbar = tqdm(total=4)
    
    # check dirs
    pbar.set_description('Checking given output and input dirs')
    check_dirs(in_dir=args.input, out_dir=args.output)
    pbar.update(1)
    
    
    # load inputs
    pbar.set_description('Make dirs')
    
    image_names = [file for file  in os.listdir(args.input) if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')) ]
    all_images = load_images(img_dir = args.input)
    pbar.update(1)
    
    # inference
    pbar.set_description('Return predictions')
    masks = inference(all_images)
    pbar.update(1)
    
    # save model prediction
    pbar.set_description('Save outputs')
    save_model_output(image_names, args.output, masks)
    pbar.update(1)
    
    
if __name__ == '__main__':
    main(args=parser.parse_args())
