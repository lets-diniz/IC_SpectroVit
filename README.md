[![GitHub](https://img.shields.io/badge/MICLab-Spectro_ViT-blue)](https://github.com/MICLab-Unicamp/Spectro-ViT)

## Project Overview

Magnetic Resonance Spectroscopy (MRS) is a non-invasive technique for quantifying metabolite concentrations in vivo, notably useful for measuring gamma-aminobutyric acid (GABA), a key inhibitory neurotransmitter in the brain. Detecting GABA is difficult due to stronger overlapping signals. MEGA-PRESS is a commonly used MRS editing method for accurate GABA measurement, although it faces challenges with low signal-to-noise ratios (SNR). To improve SNR, repeated measurements, or transients, are taken during the scan, resulting in longer scan times. Typically, scans involve 320 transients.

This repository hosts the code for the **Spectro-ViT Deep Learning (DL) model**, which **reconstructs GABA-edited MEGA-PRESS scans using only one-quarter (80 transients)** of the usual number of transients required. Spectro-ViT represents the latest advancement in this field and was the **overall winner of the Edited-MRS Reconstruction Challenge**. The repository provides a robust training framework for GABA-edited MEGA-PRESS reconstruction using the Spectro-ViT model, built on **PyTorch**. Designed for reproducibility, flexibility, and scalability, it allows researchers to easily reproduce the experiments conducted to train the Spectro-ViT model, adapt, and expand the model. Furthermore, it offers the capability to test other DL models tailored to different datasets and experimental configurations.

![qualificação-Copy of architecture](https://github.com/MICLab-Unicamp/Spectro-ViT/assets/91618118/b0cf7831-f77b-4115-bcfc-36470162d4d2)
*Figure 1: Model architecture.*

For more information about the Edited-MRS Reconstruction Challenge, visit the [challenge webpage](https://sites.google.com/view/edited-mrs-rec-challenge/home?authuser=0). Check the [journal publication](https://link.springer.com/article/10.1007/s10334-024-01156-9) documenting the challenge results. To access our trained weights and perform our model's inferences submitted in the challenge, visit the [GitHub repository from our team, Deep Spectral Divers](https://github.com/MICLab-Unicamp/MICLab-2023-ISBI-MRS-Challenge).



## Features

- **Training Framework for the Spectro-ViT Model**: Provides the training pipeline and the YAML configuration file used to develop the Spectro-ViT model, ensuring replicability and facilitating further research.
- **PyTorch Implementation**: Leverages the widely-used PyTorch library, facilitating easy integration with other machine learning tools and libraries.
- **Weights & Biases Integration**: Offers optional integration with [Weights & Biases](https://wandb.ai/site) for real-time training monitoring and results logging, enhancing the experimental tracking and analysis.
- **On-the-Fly Validation**: Supports real-time validation with visualization during training, crucial for iterative development and performance tuning. 
- **Customizable Training Framework**: Users can modify various aspects of the model and training process through the YAML configuration file, tailoring the framework to specific research needs and objectives.

## Configuration File Details

The model's training and evaluation behaviors are fully configurable through a YAML configuration file. Below, you will find detailed explanations of key sections within this file:

### Weights & Biases (wandb) Configuration

- `activate`: Enables or disables integration with Weights & Biases. Set to `True` for tracking and visualizing metrics during training.
- `project`: Specifies the project name under which the runs should be logged in Weights & Biases.
- `entity`: Specifies the user or team name under Weights & Biases where the project is located.

### Saving/Reloading Model

- **Current Model**
  - `save_model`: Enable or disable the saving of model weights.
  - `model_dir`: Directory to save the model weights.
  - `model_name`: Name under which to save the model weights.

- **Reload from Existing Model**
  - `activate`: Enable reloading weights from a previously saved model to continue training.
  - `model_dir`: Directory from where the model weights should be loaded.
  - `model_name`: Name of the model weights file to be loaded.

### Model Configuration
- **Model**
  - `Model Class Name`: Specifies the model class. Example: `SpectroViT`.
    - `Instantiation parameters of the model class`.
    
### Training Parameters

- `epochs`: Number of training epochs.
- `optimizer`: Configuration for the optimizer. Example:
  - `Adam`: Specifies using the Adam optimizer.
    - `lr`: Learning rate for the optimizer.

- `loss`: Specifies the loss function used for training. Example: `RangeMAELoss`.

- `lr_scheduler`: Configuration for the learning rate scheduler. Example:
  - `activate`: Enable or disable the learning rate scheduler.
  - `scheduler_type`: Type of scheduler, e.g., `cosineannealinglr`.
  - `info`: Parameters for the scheduler, such as `T_max` (maximum number of iterations) and `eta_min` (minimum learning rate).

### Dataset Configuration

The following configuration parameters are designed to instantiate the Dataset class:

- **Training Dataset**
  - `Dataset Class Name`: Specifies the class used for managing the training dataset. Example: `DatasetThreeChannelSpectrogram`
    - `path_data`: Directory containing the training data.
    - `evaluation`: Set to `False` to indicate training mode, which enables data augmentations.
    - `random_augment`: Configurations for random data augmentations including amplitude, frequency, and phase noise levels.

- **Validation Dataset**
  - `Dataset Class Name`: Specifies the class used for managing the validation dataset. Example: `DatasetThreeChannelSpectrogram`
    - `path_data`: Directory containing the validation data.
    - `evaluation`: Set to `True` to indicate evaluation mode.

- **Valid on the Fly**
  - `activate`: Enables saving plots that help analyze the model's performance on validation data during training.
  - `save_dir_path`: Directory to save these plots.

- **Test Dataset**
  - `Dataset Class Name`: Specifies the class used for managing the test dataset. Example: `DatasetThreeChannelSpectrogram`
    - `path_data`: Directory containing the test data.
    - `evaluation`: Set to `True` to indicate evaluation mode.

## Model Evaluation

The `evaluate.py` script evaluates the performance of the trained MRS reconstruction model. It parses command-line arguments, reads test dataset settings from the YAML file, and loads model weights. The script then processes the test dataset, generating reconstructions (predictions), and calculating reconstruction metrics such as MSE, Shape Score, SNR, and Linewidth for each sample. Additionally, the script generates and saves plots from figures 2 and 3, enabling visual inspection of the reconstruction compared to the target.

![g4_s10_slice_1](https://github.com/MICLab-Unicamp/Spectro-ViT/assets/91618118/d1a10085-686b-4750-bcc2-b9b70af387c7)
*Figure 2: Example of the plot generated by the `spectra_comparison` method from the `PlotMetrics` class*.

![g4_s10_slice_1](https://github.com/MICLab-Unicamp/Spectro-ViT/assets/91618118/a0037ce9-c420-4cf9-bdd7-d40b4648a1e7)
*Figure 3: Example of the plot generated by the `shape_score_comparison` method from the `PlotMetrics` class*.


## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/MICLab-Unicamp/Spectro-ViT.git
   ```
2. Navigate to the project directory:

   ```bash
   cd Spectro-ViT
   ```
3. Check the Python version in requirements.txt and install the required dependencies:

    ```bash
   pip install -r requirements.txt
   ```

## Training Example

Here's an example of how you might train the model using the provided configuration file:

```bash
python main.py configs/config_spectro_vit.yaml
```

## Evaluation Example

Here's an example of how you might evaluate the trained model:

```bash
python evaluate.py configs/config_spectro_vit.yaml weights/SpectroViT.pt
```

## Citation

Edited-MRS Reconstruction Challenge citation:

      @article{berto2024results,
        title={Results of the 2023 ISBI challenge to reduce GABA-edited MRS acquisition time},
        author={Berto, Rodrigo Pommot and Bugler, Hanna and Dias, Gabriel and Oliveira, Mateus and Ueda, Lucas and Dertkigil, Sergio and Costa, Paula DP and Rittner, Leticia and Merkofer, Julian P and van de Sande, Dennis MJ and others},
        journal={Magnetic Resonance Materials in Physics, Biology and Medicine},
        pages={1--15},
        year={2024},
        publisher={Springer}
      }
