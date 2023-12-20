import argparse
import os
import torch
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader
from ESRGAN import ESRGAN_g, ESRGAN_d
from auxN import TransformationPredictor
from dataset import Flickr2KDataset  # Assuming you have a custom dataset class
from train import Trainer
from sample_generator import generate_lr_images
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
from torch.utils.data import Subset


# Function to enhance an LR image
def enhance_image(lr_image_path, generator, max_size=1024):
    # Open and convert the image to tensor
    lr_image = Image.open(lr_image_path).convert("RGB")

    # Resize if the image is too large
    if lr_image.width > max_size or lr_image.height > max_size:
        lr_image = lr_image.resize((max_size, max_size), Image.ANTIALIAS)

    lr_image_tensor = ToTensor()(lr_image).unsqueeze(0)  # Add batch dimension

    # Move to the same device as the model
    device = next(generator.parameters()).device
    lr_image_tensor = lr_image_tensor.to(device)

    # Generate the high-resolution image
    with torch.no_grad():
        hr_image_tensor = generator(lr_image_tensor)

    # Convert the tensor to PIL image
    hr_image = ToPILImage()(hr_image_tensor.squeeze(0).cpu())
    return hr_image


def model_size_in_MB(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) * 4 / (1024 ** 2)


def split_dataset(full_dataset, train_size, seed=42):
    train_indices, val_indices = train_test_split(
        range(len(full_dataset)),
        train_size=train_size,
        random_state=seed
    )
    return Subset(full_dataset, train_indices), Subset(full_dataset, val_indices)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ESRGAN Model")

    # Adding arguments
    parser.add_argument('-batch', type=int, default=8, help='Batch size for training')
    parser.add_argument('-e', type=int, default=1, help='Number of epochs for training')
    parser.add_argument('-scale', type=int, default=3, help='Scale factor for the model')
    # parser.add_argument('--hrdir', type=str, required=True, help='Directory containing high resolution images')
    parser.add_argument('-hrdir', type=str, default='Flickr2K_HR', help='Directory containing high resolution images')
    parser.add_argument('-size', type=int, default=50, help='Size of the dataset to use')
    parser.add_argument('-op', type=str, default='models', help='Path to store the trained model')
    # parser.add_argument('-test_out', type=str, default='enhanced.jpg', help='Path to store the trained model')
    parser.add_argument('-check_dir', type=str, default='checkpoints', help='Path to store the trained model')

    # Parse the arguments
    args = parser.parse_args()
    # Device configuration
    torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyperparameters
    learning_rate = 0.00001
    batch_size = args.batch
    epochs = args.e
    scale_factor = args.scale
    lambda_aux = 0.1
    hr_dir = args.hrdir

    # Initialize models
    generator = nn.DataParallel(ESRGAN_g()).to(device)
    discriminator = nn.DataParallel(ESRGAN_d()).to(device)
    print("Generator Size (MB):", model_size_in_MB(generator))
    print("Discriminator Size (MB):", model_size_in_MB(discriminator))
    aux_net = TransformationPredictor(scale_factor)
    aux_net = nn.DataParallel(aux_net).to(device)
    print("Aux Net Size (MB):", model_size_in_MB(aux_net))
    # aux_net = nn.DataParallel(TransformationPredictor()).to(device)
    # print("Aux Net Size (MB):", model_size_in_MB(aux_net))

    # Optimizers
    g_optimizer = optim.Adam(generator.parameters(), lr=learning_rate)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate)
    aux_optimizer = optim.Adam(aux_net.parameters(), lr=learning_rate)

    # Loss function for the auxiliary network
    criterion = nn.MSELoss()

    # Load dataset
    full_dataset = Flickr2KDataset(hr_dir, scale_factor, transform=True)
    dataset_size = args.size  # number of images in both the training and validation sets

    train_indices = list(range(dataset_size))
    val_indices = list(range(dataset_size//4))

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    # Create DataLoaders for training and validation
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    print(f"Total training images: {len(train_dataloader.dataset)}")
    print(f"Total validation images: {len(val_dataloader.dataset)}")
    for i, real_images in enumerate(train_dataloader):
        print(f'Batch {i}: Shape - {real_images.shape}, Type - {real_images.dtype}')
        print(f'Batch {i}: Min - {real_images.min()}, Max - {real_images.max()}')

        if i == 2:
            break
    # Initialize trainer
    input_channels = next(iter(train_dataloader))[0].shape[1]
    trainer = Trainer(generator, discriminator, aux_net, g_optimizer, d_optimizer, aux_optimizer, device, input_channels, lambda_aux, scale_factor,
                      use_cuda=torch.cuda.is_available(), use_amp=True)

    # Prepare fixed low-resolution images for monitoring training progress
    fixed_lr_images = generate_lr_images(next(iter(train_dataloader)), scale_factor)

    # Train the model
    if not os.path.exists(args.check_dir):
        os.makedirs(args.check_dir)
    trainer.train(train_dataloader, val_dataloader, epochs, fixed_lr_images, args.check_dir, save_training_gif=True, validate=True)

    # Save models
    model_dir = args.op
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    g_filename = 'generator.pth'
    g_path = os.path.join(model_dir, g_filename)
    torch.save(generator.module.state_dict(), g_path)
    d_filename = 'discriminator.pth'
    d_path = os.path.join(model_dir, d_filename)
    torch.save(discriminator.module.state_dict(), d_path)
    print('Training completed and models saved.')
