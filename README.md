# IDC SinGAN: Image-to-Image Translation with Single Image GAN

## Overview
IDC SinGAN is a PyTorch implementation of a specialized Single Image Generative Adversarial Network (SinGAN). The repository focuses on performing image-to-image translation tasks like image inpainting, editing, and synthesis, all using a single input image. The ability to work with just one image makes SinGAN a versatile and powerful tool for various applications where large datasets are unavailable.

This implementation extends the capabilities of SinGAN by incorporating custom masking, multi-scale pyramid generation, and additional metrics such as accuracy for training evaluation.

---

## Features
- **Single Image GAN**: Train the model using only one input image.
- **Custom Masking**: Apply binary masks for selective image inpainting.
- **Multi-Scale Pyramid**: Adjust image scales dynamically for hierarchical training.
- **Metrics Visualization**: Log and visualize metrics such as accuracy during training.
- **Pretrained Models**: Save and load pretrained models for inference.

---

## Repository Structure
```
IDC_SinGAN/
│
├── Input/Images/        # Contains input images
│
├── SinGAN/
│   ├── models.py        # Contains the Generator and Discriminator architectures
│   ├── training.py      # Training logic for the SinGAN model
│   ├── functions.py     # Utility functions for preprocessing, mask loading, and more
│   ├── imresize.py
│   ├── manipulate.py 
│
├── main_train.py        # Entry point for training the model
├── config.py            # Configuration options and argument parsing
├── README.md            # Documentation (this file)
└── requirements.txt     # Python dependencies
```

---

## Model Architecture
### Generator
- Composed of convolutional layers with LeakyReLU activations and Tanh activation for the final output.
- Generates synthetic images from masked input.

### Discriminator
- A patch-based discriminator that distinguishes between real and generated images.
- Uses convolutional layers with BatchNorm and LeakyReLU.

### Training Procedure
1. **Discriminator Training**:
   - Trains to classify real images as real and generated images as fake.
2. **Generator Training**:
   - Trains to generate realistic images while minimizing total variation loss.
3. **Accuracy Metric**:
   - Uses mean squared error between real and fake images as an accuracy measure.

---

## Installation
### Prerequisites
- Python 3.8 or above
- CUDA-enabled GPU (recommended for faster training)

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/ShiviTripathi13/IDC_SinGAN.git
   cd IDC_SinGAN
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage
### Training the Model
1. Prepare an input image and optionally a binary mask.
2. Run the training script:
   ```bash
   python main_train.py --input_image <path_to_image> --mask <path_to_mask>
   ```
   **Example**:
   ```bash
   python main_train.py --input_image ./data/sample.png --mask ./data/mask.png
   ```

3. Outputs:
   - The model's intermediate outputs and final weights are saved in the `TrainedModels/` directory.
   - Metrics like accuracy are plotted and saved.

### Command-Line Arguments
| Argument             | Description                                   | Default              |
|----------------------|-----------------------------------------------|----------------------|
| `--input_image`      | Path to the input image                      | Required             |
| `--mask`             | Path to the binary mask                      | None (no mask)       |
| `--nfc`              | Number of feature channels in the model      | 64                   |
| `--num_layer`        | Number of layers in the model                | 5                    |
| `--ker_size`         | Kernel size for convolution layers           | 3                    |
| `--epochs`           | Number of training epochs                    | 3000                 |
| `--dir2save`         | Directory to save outputs and model weights  | `TrainedModels/`           |

### Visualizing Results
- Intermediate outputs are saved every 100 epochs in the specified `--dir2save` directory.
- After training, final outputs and metrics are also saved in the same directory.

---

## Example Flow
1. **Input Image**: Provide a single image for training.
2. **Mask Application**: Optionally apply a binary mask to the input image.
3. **Training**: Train the model to generate synthetic images.
4. **Outputs**:
   - Intermediate outputs during training (e.g., `output_epoch_100.png`).
   - Final trained Generator model (`Gs.pth`).

---

## Results
- Generated images are saved in the output directory.
- You can evaluate the performance by visualizing the generated images and metrics (accuracy, loss).

---

## Contributing
We welcome contributions to enhance the features and performance of IDC SinGAN. Feel free to submit pull requests or open issues for discussions.

---

## License
This repository is licensed under the [MIT License](LICENSE).

---

## Acknowledgements
This project is inspired by the original SinGAN paper. Special thanks to the contributors of open-source libraries like PyTorch and Matplotlib.

---

For further details, please contact [ShiviTripathi13](https://github.com/ShiviTripathi13).
