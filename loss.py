import torch
import torch.nn as nn
from torchvision.models.vgg import vgg16
import torch.nn.functional as F


class VGGFeatureExtractor(nn.Module):
    def __init__(self):
        super(VGGFeatureExtractor, self).__init__()
        vgg = vgg16(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, img):
        return self.feature_extractor(img)


class GANLoss(nn.Module):
    def __init__(self, feature_extractor, in_channels=3, lambda_adv=0.001, lambda_percep=0.006, lambda_mse=1.0,
                 lambda_tv=2e-8, lambda_ssim=0.1):
        super(GANLoss, self).__init__()
        self.feature_extractor = feature_extractor
        self.mse_loss = nn.MSELoss()
        self.tv_loss = TVLoss()
        # self.ssim_loss = SSIMLoss(in_channels)
        self.lambda_adv = lambda_adv
        self.lambda_percep = lambda_percep
        self.lambda_mse = lambda_mse
        self.lambda_tv = lambda_tv
        self.lambda_ssim = lambda_ssim

    def forward(self, out_labels, out_images, target_images):
        out_images_resized = F.interpolate(out_images, size=(target_images.size(2), target_images.size(3)),
                                           mode='bilinear', align_corners=False)

        adversarial_loss = torch.mean(1 - out_labels)
        perception_loss = self.mse_loss(self.feature_extractor(out_images_resized),
                                        self.feature_extractor(target_images))
        image_loss = self.mse_loss(out_images_resized, target_images)
        tv_loss = self.tv_loss(out_images_resized)
        # ssim_loss = self.ssim_loss(out_images_resized, target_images)

        # Aggregate losses
        total_loss = (self.lambda_mse * image_loss +
                      self.lambda_adv * adversarial_loss +
                      self.lambda_percep * perception_loss +
                      self.lambda_tv * tv_loss)
        return total_loss


class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]


if __name__ == "__main__":
    vgg_extractor = VGGFeatureExtractor()
    gan_loss = GANLoss(vgg_extractor)
    print(gan_loss)
