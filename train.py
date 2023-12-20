import os
import imageio
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from torch import nn
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
from sample_generator import generate_lr_images
from torch.cuda.amp import GradScaler, autocast
from loss import VGGFeatureExtractor, GANLoss  # Import the GANLoss
import pytorch_ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np


class Trainer:
    def __init__(self, generator, discriminator, auxN, gen_optimizer, dis_optimizer, aux_optimizer, device, ic, lambda_aux=0.1, scale_factor=4, use_cuda=True, use_amp=False):
        self.losses = {'G': [], 'D': []}
        self.G = generator
        self.D = discriminator
        self.G_opt = gen_optimizer
        self.D_opt = dis_optimizer
        self.Aux = auxN
        self.Aux_opt = aux_optimizer
        self.lambda_aux = lambda_aux
        self.scale_factor = scale_factor
        self.use_cuda = use_cuda
        self.device = device
        self.num_steps = 0
        self.print_every = 50

        self.vgg_extractor = VGGFeatureExtractor().to(self.device)
        self.gan_loss = GANLoss(self.vgg_extractor, in_channels=ic).to(self.device)

        if self.use_cuda:
            self.G = generator.to(self.device)
            self.G = nn.DataParallel(self.G).cuda()
            self.D = discriminator.to(self.device)
            self.D = nn.DataParallel(self.D).cuda()
            self.Aux = auxN.to(self.device)
            self.Aux = nn.DataParallel(self.Aux).cuda()
        self.scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    def _train_discriminator(self, real_images, fake_images):
        self.D_opt.zero_grad()

        real_predictions = self.D(real_images)
        fake_predictions = self.D(fake_images.detach())

        # Calculate losses for real and fake images
        real_loss = F.binary_cross_entropy_with_logits(real_predictions, torch.ones_like(real_predictions))
        fake_loss = F.binary_cross_entropy_with_logits(fake_predictions, torch.zeros_like(fake_predictions))

        total_loss = real_loss + fake_loss
        total_loss.backward()

        self.D_opt.step()

        self.losses['D'].append(total_loss.item())


    def auxn_loss(self, fake_images, real_images):
        predicted_transformations = self.Aux(fake_images)

        actual_transformations = torch.full_like(predicted_transformations, self.scale_factor)

        loss = F.mse_loss(predicted_transformations, actual_transformations)

        return loss

    def _train_auxiliary_network(self, hr_images, lr_images):

        hr_images = hr_images.to(self.device)
        lr_images = lr_images.to(self.device)

        predicted_lr_images = self.Aux(hr_images)
        predicted_lr_images = predicted_lr_images.to(self.device)
        # Print shapes for debugging
        # print("Predicted Transformations Shape:", predicted_lr_images.shape)
        # print("HR Images Shape:", hr_images.shape)
        # print("LR Images Shape:", lr_images.shape)
        loss = F.mse_loss(predicted_lr_images, lr_images)

        self.Aux_opt.zero_grad()
        loss.backward()
        self.Aux_opt.step()
        return loss.item()

    def _train_generator_with_auxn_feedback(self, fake_images, real_images, lr_images):
        self.G_opt.zero_grad()

        fake_predictions = self.D(fake_images)

        # Calculate GAN loss using binary_cross_entropy_with_logits
        # gan_loss = F.binary_cross_entropy_with_logits(fake_predictions, torch.ones_like(fake_predictions))
        gan_loss = self.gan_loss(fake_predictions, fake_images, real_images)

        # Calculate AuxN loss
        auxn_loss = self.auxn_loss(fake_images, real_images)

        combined_loss = gan_loss + self.lambda_aux * auxn_loss

        combined_loss.backward()
        self.G_opt.step()

        self.losses['G'].append(combined_loss.item())


    def calculate_psnr(self, fake_images, real_images):
        fake_images_np = fake_images.cpu().detach().numpy()
        real_images_np = real_images.cpu().detach().numpy()

        psnr_sum = 0.0

        # Iterate over batch
        for i in range(fake_images_np.shape[0]):
            psnr_value = psnr(real_images_np[i].transpose(1, 2, 0), fake_images_np[i].transpose(1, 2, 0),
                              data_range=real_images_np[i].max() - real_images_np[i].min())
            psnr_sum += psnr_value

        # Calculate average PSNR
        average_psnr = psnr_sum / fake_images_np.shape[0]
        return average_psnr


    def validate(self, val_dataloader):
        self.G.eval()
        self.D.eval()
        val_losses = {'G': [], 'D': []}
        additional_metrics = {'PSNR': []}

        with torch.no_grad():  # Disable gradient calculation
            for real_images in val_dataloader:
                real_images = real_images.to(self.device)
                sample_input = generate_lr_images(real_images, self.scale_factor).to(self.device)

                fake_images = self.G(sample_input)
                fake_images = F.interpolate(fake_images, size=(real_images.size(2), real_images.size(3)),
                                            mode='bilinear', align_corners=False)
                fake_predictions = self.D(fake_images)

                gan_loss = F.binary_cross_entropy(fake_predictions, torch.ones_like(fake_predictions))
                auxn_loss = self.auxn_loss(fake_images, real_images)
                combined_loss = gan_loss + self.lambda_aux * auxn_loss

                val_losses['G'].append(combined_loss.item())

                real_predictions = self.D(real_images)
                fake_predictions = self.D(fake_images)
                real_loss = F.binary_cross_entropy_with_logits(real_predictions, torch.ones_like(real_predictions))
                fake_loss = F.binary_cross_entropy_with_logits(fake_predictions, torch.zeros_like(fake_predictions))
                d_loss = real_loss + fake_loss

                val_losses['D'].append(d_loss.item())

                # ssim_value = self.calculate_ssim(fake_images, real_images)
                psnr_value = self.calculate_psnr(fake_images, real_images)
                # additional_metrics['SSIM'].append(ssim_value)
                additional_metrics['PSNR'].append(psnr_value)

            self.G.train()
            self.D.train()

            avg_val_losses = {key: np.mean(val_losses[key]) for key in val_losses}
            avg_psnr_losses = {key: np.mean(additional_metrics[key]) for key in additional_metrics}

        return avg_val_losses, avg_psnr_losses

    def train_epoch(self, dataloader, epoch):
        scaler = GradScaler()

        for i, real_images in enumerate(dataloader):
            real_images = real_images.to(self.device)
            # print("real Images Shape:", real_images.shape)

            sample_input = generate_lr_images(real_images, self.scale_factor).to(self.device)
            # print("sample input Shape:", sample_input.shape)

            # train the auxiliary network
            self._train_auxiliary_network(real_images, sample_input)

            # train the generator and discriminator
            torch.cuda.empty_cache()
            with autocast():  # Mixed precision
                fake_images = self.G(sample_input)
                # print("fake input Shape:", fake_images.shape)
                # Train discriminator
                fake_images = F.interpolate(fake_images, size=(real_images.size(2), real_images.size(3)),
                                            mode='bilinear', align_corners=False)

                self._train_discriminator(real_images, fake_images)

                # Train generator with feedback from AuxN
                fake_images = self.G(sample_input)  # re-generate to avoid stale images
                fake_images = F.interpolate(fake_images, size=(real_images.size(2), real_images.size(3)),
                                            mode='bilinear', align_corners=False)

                # print("fake input 2 Shape:", fake_images.shape)
                self._train_generator_with_auxn_feedback(fake_images, real_images, sample_input)

            if self.num_steps % self.print_every == 0:
                print(f'Batch {i}: Discriminator Loss: {self.losses["D"][-1]}, Generator Loss: {self.losses["G"][-1]}')
            self.num_steps += 1

            if self.num_steps % self.print_every == 0:
                print(f'Discriminator Loss: {self.losses["D"][-1]}, Generator Loss: {self.losses["G"][-1]}')
            self.num_steps += 1


    def train(self, dataloader, val_dataloader, epochs, fixed_lr_images, check_dir,
              save_training_gif=True, validate=False):  # fixed lr images for monitoring training process
        training_progress_images = []
        training_progress_images = []
        g_losses = []
        d_losses = []
        val_g_losses = []
        val_d_losses = []
        psnr = []
        checkpoint_dir = check_dir
        best_ssim = -float('inf')
        best_psnr = -float('inf')

        # Early stopping parameters
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 500


        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            self.train_epoch(dataloader, epoch)

            if validate:
                val_losses, metrics = self.validate(val_dataloader)
                val_g_losses.append(val_losses['G'])
                val_d_losses.append(val_losses['D'])
                print(f'Validation: Discriminator Loss: {val_losses["D"]}, Generator Loss: {val_losses["G"]}')

                # Check for improvement
                current_val_loss = val_losses['G']

                # ssim = metrics['SSIM'][-1]
                psnr.append(metrics['PSNR'])
                if current_val_loss < best_val_loss or metrics['PSNR'] > best_psnr:
                    best_psnr = metrics['PSNR']
                    best_val_loss = current_val_loss
                    patience_counter = 0
                    best_path = os.path.join(checkpoint_dir, 'best_generator.pth')
                    torch.save(self.G.module.state_dict(), best_path)
                    print(f"Epoch {epoch + 1}: Training loss improved to {best_val_loss:.4f}. Model saved.")
                else:
                    patience_counter += 1

                if patience_counter > 0:
                    print(f"Validation loss has not improved for {patience_counter} epochs.")
                if patience_counter >= patience:
                    print("Early stopping triggered after validation loss has stopped improving.")
                    break

                # Save checkpoint
                if epoch % 5 == 0:
                    if not os.path.exists(checkpoint_dir):
                        os.makedirs(checkpoint_dir)
                    checkpoint_filename = f'checkpoint_epoch_{epoch}.pth'
                    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
                    torch.save(self.G.module.state_dict(), checkpoint_path)

            if save_training_gif:
                fixed_hr_images = self.G(fixed_lr_images.to(self.device))
                fixed_hr_images = fixed_hr_images.cpu().detach()  # Detach from the computation graph

                img_grid = make_grid(fixed_hr_images)

                img_grid = (img_grid - img_grid.min()) / (img_grid.max() - img_grid.min())
                img_grid = (img_grid.numpy() * 255).astype(np.uint8)

                img_grid = np.transpose(img_grid, (1, 2, 0))
                training_progress_images.append(img_grid)

            g_losses.append(np.mean(self.losses['G'][-len(dataloader):]))
            d_losses.append(np.mean(self.losses['D'][-len(dataloader):]))

            self.plot_and_save(g_losses, 'G Train Loss', checkpoint_dir, 'train_loss.png',
                               "Generator and Discriminator Loss During Training", 'Loss',
                               d_losses, 'D Train Loss')

            if validate:
                self.plot_and_save(val_g_losses, 'Generator Val Loss', checkpoint_dir, 'validation_loss.png',
                                   "Validation Loss During Training", 'Loss',
                                   val_d_losses, 'Discriminator Val Loss')
                self.plot_and_save(psnr, "PSNR During Training", checkpoint_dir, 'psnr.png',
                                   "PSNR During Training", 'PSNR')
            if save_training_gif:
                gif_path = os.path.join(checkpoint_dir, 'training_progress.gif')
                imageio.mimsave(gif_path, training_progress_images, fps=2)

    def plot_and_save(self, value1, label1, check_dir, title, name, ylabel=None, value2=None, label2=None):
        plt.figure(figsize=(10, 5))
        plt.title(title)
        plt.plot(value1, label=label1)
        if value2:
            plt.plot(value2, label=label2)
        plt.xlabel("Epochs")
        plt.ylabel(ylabel)
        plt.legend()
        path = os.path.join(check_dir, name)
        plt.savefig(path)
        plt.close()
# Example usage:
# fixed_lr_images = [code to generate/select fixed low-resolution images]
# trainer = Trainer(generator, discriminator, g_optimizer, d_optimizer, use_cuda=torch.cuda.is_available())
# trainer.train(dataloader, epochs, fixed_lr_images)