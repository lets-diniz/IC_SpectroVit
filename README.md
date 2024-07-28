[![GitHub](https://img.shields.io/badge/MICLab-Spectro_ViT-red)](https://github.com/MICLab-Unicamp/Spectro-ViT) [![DOI](https://img.shields.io/badge/DOI-10.1016/j.mri.2024.110219-blue)](https://doi.org/10.1016/j.mri.2024.110219)


## Project Overview

Magnetic Resonance Spectroscopy (MRS) is a non-invasive technique for quantifying metabolite concentrations in vivo, notably useful for measuring gamma-aminobutyric acid (GABA), a key inhibitory neurotransmitter in the brain. Detecting GABA is difficult due to stronger overlapping signals. MEGA-PRESS is a commonly used MRS editing method for accurate GABA measurement, although it faces challenges with low signal-to-noise ratios (SNR). To improve SNR, repeated measurements, or transients, are taken during the scan, resulting in longer scan times. Typically, scans involve 320 transients.

This repository hosts the code for the Spectro-ViT Deep Learning (DL) model, which reconstructs GABA-edited MEGA-PRESS scans using only one-quarter (80 transients) of the usual 320 transients required. Spectro-ViT represents the latest advancement in this field and was the overall winner of the Edited-MRS Reconstruction Challenge. The repository provides a robust training framework for GABA-edited MEGA-PRESS reconstruction using the Spectro-ViT model, built on PyTorch. Designed for reproducibility, flexibility, and scalability, it allows researchers to easily reproduce the experiments conducted to train the Spectro-ViT model, adapt, and expand the model. Furthermore, it offers the capability to test other DL models tailored to different datasets and experimental configurations.

![Captura de tela de 2024-07-28 10-55-12](https://github.com/user-attachments/assets/09f31a54-1f85-4b8d-8273-13ab05feb572)


*Figure 1: Model architecture. Source: https://doi.org/10.1016/j.mri.2024.110219*

Our work on the Spectro-ViT model has been published in the Magnetic Resonance Imaging journal. You can find the full publication at this [DOI link](https://doi.org/10.1016/j.mri.2024.110219).

For more information about the Edited-MRS Reconstruction Challenge, visit the [challenge webpage](https://sites.google.com/view/edited-mrs-rec-challenge/home?authuser=0). Check the [journal publication](https://link.springer.com/article/10.1007/s10334-024-01156-9) documenting the challenge results. 


## Features

- **Training Framework for the Spectro-ViT Model**: Provides the training pipeline and the YAML configuration file used to develop the Spectro-ViT model, ensuring replicability and facilitating further research.
- **PyTorch Implementation**: Leverages the widely-used PyTorch library, facilitating easy integration with other machine learning tools and libraries.
- **Weights & Biases Integration**: Offers optional integration with [Weights & Biases](https://wandb.ai/site) for real-time training monitoring and results logging, enhancing the experimental tracking and analysis.
- **On-the-Fly Validation**: Supports real-time validation with visualization during training, crucial for iterative development and performance tuning. 
- **Customizable Training Framework**: Users can modify various aspects of the model and training process through the YAML configuration file, tailoring the framework to specific research needs and objectives.

## HDF5 File Data Organization

The data is expected to be organized in an HDF5 file with the following keys:

- **transient_specs**: Contains the transients as a NumPy array with the shape (number of points, subsignal axis, number of transients). The subsignal axis 0 is for the edit-off signals and axis 1 for the edit-on signals.
- **target_spectra**: Contains the target spectrum as a NumPy array.
- **ppm**: Parts per million (ppm) scale as a NumPy array.
- **fs**: Sampling frequency as a float64.
- **tacq**: Acquisition time as a float64.
- **larmorfreq**: Spectrometer frequency as a float64.

The `read_h5_complete` function reads these keys from the specified HDF5 file and returns them as a tuple of NumPy arrays and float64 values.

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

## Developers

- [Gabriel Dias](https://github.com/gsantosmdias)
- [Mateus Oliveira](https://github.com/oliveiraMats2)
- [Lucas Ueda](https://github.com/lucashueda)

## Credits
- The _in-vivo_ data utilized is this study was sampled from the [Big GABA repository](https://www.nitrc.org/projects/biggaba/) ([Publication](https://pubmed.ncbi.nlm.nih.gov/28716717/)).
- The target data and quantification results were obtained using the [Gannet software](https://onlinelibrary.wiley.com/doi/full/10.1002/jmri.24478): [![Gannet](https://img.shields.io/badge/markmikkelsen-Gannet-orange)](https://github.com/markmikkelsen/Gannet).

- The Data Augmentation code used in this research was adapted from: [![GitHub](https://img.shields.io/badge/rmsouza01-Edit_MRS_Challenge-purple)](https://github.com/rmsouza01/Edited-MRS-challenge).

- This research utilized Python code that interfaces with MATLAB to quantify the reconstructed spectra using Gannet, derived from: [![rodrigopberto](https://img.shields.io/badge/rodrigopberto-Edited%20MRS%20DL%20Reconstruction-green)](https://github.com/rodrigopberto/Edited-MRS-DL-Reconstruction).



## Citation

    @article{DIAS2024,
    title = {Spectro-ViT: A vision transformer model for GABA-edited MEGA-PRESS reconstruction using spectrograms},
    journal = {Magnetic Resonance Imaging},
    pages = {110219},
    year = {2024},
    doi = {https://doi.org/10.1016/j.mri.2024.110219},
    author = {Gabriel Dias and Rodrigo Pommot Berto and Mateus Oliveira and Lucas Ueda and Sergio Dertkigil and Paula D.P. Costa and Amirmohammad Shamaei and Hanna Bugler and Roberto Souza and Ashley Harris and Leticia Rittner}
    }
