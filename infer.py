import os
import argparse
import torch
from torch import nn
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torchvision.transforms import ToTensor, ToPILImage
#from ESRGAN import ESRGAN_g
from ESRGAN import ESRGAN_g


def infer(lr_image, generator, max_size=1024):
    old_size = lr_image.size

    # Respect original ratio
    ratio = float(max_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    if lr_image.width > max_size or lr_image.height > max_size:
        lr_image = lr_image.resize(new_size, Image.Resampling.BICUBIC)

    lr_image_tensor = ToTensor()(lr_image).unsqueeze(0)  # Add batch dimension

    # Move to the same device as the model
    device = next(generator.parameters()).device
    lr_image_tensor = lr_image_tensor.to(device)

    with torch.no_grad():
        hr_image_tensor = generator(lr_image_tensor)

    hr_image = ToPILImage()(hr_image_tensor.squeeze(0).cpu())
    return hr_image


def load_model(model_path, device):
    generator = ESRGAN_g().to(device)
    generator = nn.DataParallel(generator)
    if os.path.isfile(model_path):
        state_dict = torch.load(model_path, map_location=device)
        generator.load_state_dict(state_dict)
        print("Loaded model from '{}'".format(model_path))
    else:
        print("No model found at '{}'".format(model_path))
        exit()
    return generator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhance low resolution images using ESRGAN")
    parser.add_argument("-model", type=str, required=True, help="Path to the trained SRGAN model")
    parser.add_argument("-input", type=str, required=True, help="Path to the input low resolution image")
    parser.add_argument("-output", type=str, required=True, help="Path to save the enhanced image")
    parser.add_argument("-t", type=int, default=1, help="Times to infer")

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = load_model(args.model, device)

    # best_model_path = '500e100p_mod_j/best_generator.pth'
    # best_model_path = 'test/best_generator.pth'
    # best_model_path = '1000e1000p_org/best_generator.pth'

    # hr_image = enhance_image("test_lr/000001x3.png", generator)  # path to lr img
    # hr_image = enhance_image("test_lr/test.png", generator)  # path to lr img
    lr_image = Image.open(args.input).convert("RGB")
    hr_image = infer(lr_image, generator)
    print("Loaded image from '{}'".format(args.input))
    if args.t != 1:
        for i in range(args.t-1):
            hr_image = infer(hr_image, generator)
    # hr_image.save("enhanced_image_org.jpg")
    hr_image.save(args.output)
    print(f"Enhanced image saved to {args.output}")
