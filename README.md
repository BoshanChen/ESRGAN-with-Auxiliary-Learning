# ESRGAN-with-Auxiliary-Learning
## Project Overview

Our ESRGAN implementation builds upon the original ESRGAN architecture by integrating a self-attention mechanism and an auxiliary network. The self-attention module helps capture global dependencies within images, enhancing detail accuracy and overall image quality. The auxiliary network is designed to learn the transformation from high-resolution (HR) to low-resolution (LR) images, providing additional feedback for the generator.

## Features

- **Self-Attention Mechanism**: Enhances the model's ability to focus on global contextual information within images.
- **Auxiliary Network**: Assists in learning the downscaling process from HR to LR images, improving the generator's performance.
- **Advanced Loss Functions**: Incorporates a combination of adversarial loss, perceptual loss, and total variation loss for comprehensive training.

#### To train the model, run the following command

```Bash
python train.py --train-dir ./dataset/train/ --val-dir ./dataset/val/

```

#### To enhance image quality, run the following command

```Bash
python train.py --train-dir ./dataset/train/ --val-dir ./dataset/val/
```
