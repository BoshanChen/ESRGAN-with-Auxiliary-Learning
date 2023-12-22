# ESRGAN-with-Auxiliary-Learning

## Project Overview

The ESRGAN implementation builds upon the original ESRGAN architecture by integrating a self-attention mechanism and an auxiliary network. The self-attention module helps capture global dependencies within images, enhancing detail accuracy and overall image quality. The auxiliary network is designed to learn the transformation from high-resolution (HR) to low-resolution (LR) images, providing additional feedback for the generator.

## Dependencies

- torch==2.1.2+cu121
- scikit-image==0.22.0
- Pillow==10.0.1
- matplotlib==3.8.0

## Dataset

This project uses the Flickr2K_HR dataset for training and evaluation.

### Flickr2K Dataset Overview

- **Source**: The Flickr2K_HR dataset is a high-quality collection of 2,000 2K resolution images sourced from Flickr. It is widely used in image super-resolution tasks.
- **Contents**: The dataset includes a diverse range of images, making it suitable for training models that require high-resolution image data.

## Features

- **Self-Attention Mechanism**: Enhances the model's ability to focus on global contextual information within images.
- **Auxiliary Network**: Assists in learning the downscaling process from HR to LR images, improving the generator's performance.

#### To train the model, run the following command

```Bash
python main.py [-batch] batch_size -e epochs [-scale] scale_factor -hrdir dir_hr_imgs [-size] size_of_dataset_to_use -op output_path -check_dir checkpoints_path

```

#### To enhance image quality with trained model, run the following command

```Bash
python infer.py -model path_to_model -input input_path -output output_path -t iterations_to_enhance
```

