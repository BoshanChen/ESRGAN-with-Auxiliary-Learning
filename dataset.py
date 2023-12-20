import argparse
import os
import random
from PIL import Image, ImageFilter
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Resize
from PIL import Image
from torchvision import transforms

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, required=True, help="Directory with the dataset")
parser.add_argument('--output_dir', type=str, required=True, help="Where to write the new data")
parser.add_argument('--hr_size', type=int, default=256, help="High-resolution image size")
parser.add_argument('--lr_size', type=int, default=64, help="Low-resolution image size")
parser.add_argument('--file_ext', type=str, default='png', help="File extension of images")
parser.add_argument('--augment', action='store_true', help="Apply data augmentation")

class Flickr2KDataset(Dataset):
    def __init__(self, hr_dir, scale_factor, transform=True):
        self.hr_dir = hr_dir
        self.scale_factor = scale_factor
        self.hr_images = os.listdir(hr_dir)
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),  # Resize to a fixed size
            transforms.ToTensor()
        ]) if transform else None

    def __len__(self):
        return len(self.hr_images)

    def __getitem__(self, idx):
        hr_image_path = os.path.join(self.hr_dir, self.hr_images[idx])
        hr_image = Image.open(hr_image_path).convert('RGB')

        if self.transform:
            hr_image = self.transform(hr_image)

        return hr_image


def random_crop(img, crop_size):
    width, height = img.size
    rand_x = random.randint(0, width - crop_size)
    rand_y = random.randint(0, height - crop_size)
    return img.crop((rand_x, rand_y, rand_x + crop_size, rand_y + crop_size))


def process_and_save(filename, output_dir_hr, output_dir_lr, hr_size, lr_size, augment):
    image = Image.open(filename).convert("RGB")

    if augment:
        image = random_crop(image, hr_size)
        if random.random() > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() > 0.5:
            image = image.filter(ImageFilter.GaussianBlur(radius=1))

    hr_image = image.resize((hr_size, hr_size), Image.BICUBIC)
    lr_image = hr_image.resize((lr_size, lr_size), Image.BICUBIC)

    base_filename = os.path.splitext(os.path.basename(filename))[0]
    hr_image.save(os.path.join(output_dir_hr, f"{base_filename}.{args.file_ext}"))
    lr_image.save(os.path.join(output_dir_lr, f"{base_filename}.{args.file_ext}"))


if __name__ == '__main__':
    args = parser.parse_args()

    assert os.path.isdir(args.data_dir), f"Couldn't find the dataset at {args.data_dir}"

    output_dir_hr = os.path.join(args.output_dir, 'HR')
    output_dir_lr = os.path.join(args.output_dir, 'LR')
    os.makedirs(output_dir_hr, exist_ok=True)
    os.makedirs(output_dir_lr, exist_ok=True)

    filenames = [os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir) if f.endswith(f'.{args.file_ext}')]

    print(f"Processing images, saving HR to {output_dir_hr} and LR to {output_dir_lr}")

    for filename in tqdm(filenames):
        process_and_save(filename, output_dir_hr, output_dir_lr, args.hr_size, args.lr_size, args.augment)

    print("Done building dataset")
