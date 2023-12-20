from PIL import Image
import torchvision.transforms as transforms
import torch


def generate_lr_images(hr_images_batch, scale_factor=4):
    lr_images_batch = []

    to_pil = transforms.ToPILImage()
    to_tensor = transforms.ToTensor()

    for hr_image_tensor in hr_images_batch:
        # Convert tensor to PIL Image
        hr_image = to_pil(hr_image_tensor)

        # Resize image
        width, height = hr_image.size
        lr_image = hr_image.resize((width // scale_factor, height // scale_factor), Image.BICUBIC)

        # Convert back to tensor
        lr_image_tensor = to_tensor(lr_image)
        lr_images_batch.append(lr_image_tensor)

    # Stack all the LR image tensors
    lr_images_batch = torch.stack(lr_images_batch)

    return lr_images_batch
