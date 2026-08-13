# SZ-hiPSC-DHM-Calcium-UNet

## Multimodal segmentation and analysis of schizophrenia hiPSC-derived neurons using DHM, calcium imaging, and U-Net

U-Net Cell Body and U-Net Neurite are deep-learning models developed to identify neuronal cell bodies and neurites in multimodal images of schizophrenia (SZ) and healthy control (CTL) hiPSC-derived neuronal cultures.

Both U-Net models were trained using digital holographic microscopy (DHM) quantitative phase images and calcium fluorescence images. The multimodal image sequences are processed into projection images and divided into image patches before being provided to the corresponding U-Net segmentation pipelines.

The trained models generate probability maps of neuronal cell bodies and neurites. These segmentation results are used to identify individual neurons, quantify neuronal morphology, and extract matched quantitative phase and calcium fluorescence time series.

The resulting morphological, biophysical, and functional features are used to investigate maturation-dependent differences between CTL and SZ neuronal cultures and to classify their disease-state identity.

- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)

The U-Net Cell Body and U-Net Neurite architectures are based on an implementation of the [U-Net model for Keras](https://github.com/pietz/unet-keras).

![Multimodal hiPSC-derived neuronal image-analysis pipeline](hIPSC-pipeline.png "DHM and Calcium Imaging Analysis Pipeline")

## Overview

### Multimodal Image Acquisition

DHM quantitative phase and calcium fluorescence image sequences are acquired from the same hiPSC-derived neuronal cultures at a frequency of 1 Hz.

Recordings are performed under three pharmacological conditions:

- Baseline receptor activity
- Selective non-NMDAR blockade
- Combined non-NMDAR and NMDAR blockade

After an initial period of spontaneous activity, the neuronal cultures are stimulated with glutamate. Quantitative phase and calcium signals are recorded to characterize the corresponding biophysical and functional responses.

### Image Projection and Preprocessing

The acquired DHM and calcium image sequences are converted into two-dimensional projection images before segmentation:

- Average-projection images are generated from the DHM quantitative phase sequences.
- Maximum-projection images are generated from the calcium fluorescence sequences.

The projection images are divided into smaller image patches, which are provided as inputs to the U-Net segmentation models. After prediction, the segmented patches are merged to reconstruct probability maps covering the complete field of view.

### U-Net Cell Body

U-Net Cell Body is a deep-learning model trained using multimodal DHM quantitative phase and calcium fluorescence images to identify neuronal cell bodies.

The model generates cell-body probability maps that are used to:

- Identify individual neurons
- Determine cell-body positions
- Measure cell-body area
- Calculate neuronal density
- Define cellular regions of interest
- Extract matched quantitative phase time series
- Extract matched calcium fluorescence time series

### U-Net Neurite

U-Net Neurite is a deep-learning model trained using multimodal DHM quantitative phase and calcium fluorescence images to identify neuronal processes.

The model generates neurite probability maps that are used to:

- Identify neuronal processes
- Measure neurite length
- Quantify neurite distribution
- Determine probable connections between neuronal cell bodies
- Reconstruct structural neuronal networks

### Quantitative Phase Time-Series Extraction

The cell-body probability maps are applied to the DHM image sequences to extract quantitative phase time series for individual neurons.

The phase signals are corrected and normalized relative to their prestimulation baselines. These signals provide information about glutamate-evoked biophysical changes associated with:

- Intracellular mass redistribution
- Variations in cellular optical path length
- Changes in cell thickness
- Transmembrane water movements
- Volume-regulatory processes

### Calcium Time-Series Extraction

The cell-body probability maps are also applied to the calcium fluorescence image sequences to extract calcium-associated time series for individual neurons.

After baseline correction, changes in fluorescence intensity are used to quantify glutamate-evoked intracellular calcium responses under the different pharmacological conditions.

### Multimodal Feature Extraction

The segmentation masks and extracted time series are combined to calculate:

- Cell-body area
- Neurite length
- Neuronal density
- Quantitative phase features
- Calcium fluorescence features
- Phase-response dynamics
- Calcium-response dynamics
- Structural network properties
- Functional network properties
- Graph-theoretical and spectral features
- Maturation-dependent neuronal signatures

### Disease-State Classification

The extracted morphological, phase-derived, calcium-associated, and network features are used to identify differences between CTL and SZ hiPSC-derived neuronal cultures.

Separate Random Forest classifiers are developed for week 2 and week 6 because the most informative disease-associated features may change during neuronal maturation. The predictions from both maturation stages are subsequently integrated to classify each cell line as CTL or SZ.
### Disease-State Classification

Morphological, phase-derived, calcium-associated, and network features are combined using machine-learning methods to classify healthy control and schizophrenia hiPSC-derived neuronal cultures.

Separate Random Forest classifiers are developed for week 2 and week 6, followed by an integrated maturation-stage classification framework.

## Dependencies

Python 3.11, TensorFlow, Keras, PyTorch, OpenCV, scikit-image, scikit-learn, pandas, NumPy, SciPy, statsmodels, Matplotlib, Seaborn, and NetworkX.

## Installation

1. Clone this repository.

```bash
git clone https://github.com/zyazdani-92/SZ-hiPSC-DHM-Calcium-UNet.git
cd SZ-hiPSC-DHM-Calcium-UNet
```

2. Create a virtual environment.

```bash
conda create -n SZ-hiPSC python=3.11
conda activate SZ-hiPSC
```

3. Install the dependencies.

```bash
pip install -r requirements.txt
```

***

## Training

The U-Net Cell Body and U-Net Neurite models are trained using the [U-Net training pipeline](https://github.com/zyazdani-92/UDHM-Cellbody-Neurite/tree/main/U-Net-training-pipeline) available in the [UDHM-Cellbody-Neurite repository](https://github.com/zyazdani-92/UDHM-Cellbody-Neurite).

Two separate U-Net models are trained:

- **U-Net Cell Body** is trained using DHM quantitative phase and calcium fluorescence images with their corresponding manually annotated cell-body masks.
- **U-Net Neurite** is trained using DHM quantitative phase and calcium fluorescence images with their corresponding manually annotated neurite masks.

1. Clone the training repository:

```bash
git clone https://github.com/zyazdani-92/UDHM-Cellbody-Neurite.git
cd UDHM-Cellbody-Neurite/U-Net-training-pipeline
```

2. Place the training and validation images and their corresponding manually annotated masks in the appropriate directories inside the `data` folder.

3. Run the `trainUNet.ipynb` notebook using the cell-body dataset to train the U-Net Cell Body model.

4. Repeat the training procedure using the neurite dataset to train the U-Net Neurite model.

The trained models used in this project are:

- Cell-body model: `DF-model.hdf5`
- Neurite model: `DHM-Fluo_neurite_model.hdf5`

The complete patient-derived imaging dataset is not publicly available in this repository because of data-access restrictions.

***

## Brief Usage Guidelines for the Trained Models

1. Perform motion correction on the DHM quantitative phase and calcium fluorescence image sequences.

2. Register the calcium fluorescence images with the corresponding DHM images.

3. Generate average-projection images from the DHM image sequences and maximum-projection images from the calcium fluorescence sequences.

4. Place the projection images in the `Test_img` folder.

5. Load the following trained models:

   - `DF-model.hdf5` for cell-body segmentation
   - `DHM-Fluo_neurite_model.hdf5` for neurite segmentation

6. Run the `Img2map_pipeline.ipynb` notebook located in the prediction pipeline folder.

7. Generate the cell-body and neurite probability maps.

8. Apply the optimized thresholds and watershed segmentation to produce binary neurite masks and labeled cell-body masks.

9. Combine the segmentation outputs to quantify cell-body morphology, neurite organization, and structural network properties.

10. Apply the cell-body masks to the registered DHM and calcium image sequences to extract matched quantitative phase and calcium fluorescence time series.

11. Extract morphological, phase-derived, calcium-associated, graph-theoretical, and network features.

12. Use the extracted multimodal features for statistical analysis and CTL/SZ disease-state classification.


***

## Data Availability

The raw data supporting this study contain patient-derived imaging information and are not publicly distributed through this repository.

The data may be made available by the corresponding author upon reasonable request and subject to applicable ethical, institutional, and data-sharing requirements.

***

## About

This package is part of a Ph.D. project conducted by [Zahra Yazdani](https://github.com/zyazdani-92) under the supervision of [Patrick Desrosiers](https://github.com/pdesrosiers) and [Antoine Allard](https://github.com/antoineallard) at the [Dynamica Research Lab](https://github.com/DynamicaLab), Université Laval.

This research is conducted in collaboration with [Pierre Marquet](https://scholar.google.ca/citations?user=-hYR_owAAAAJ&hl=en&oi=sra) at the [Laboratoire de recherche en neurophotonique et psychiatrie](https://www.labrnp.ca/) at the CERVO Brain Research Centre. The hiPSC-derived neuronal cultures were prepared and imaged by [Niraj Patel](https://orcid.org/0000-0002-7820-202X) in Pierre Marquet’s laboratory.

This research is part of the Neuro-CERVO Alliance for Drug Discovery in Brain Diseases (NCADD), which supports collaborative research into the mechanisms and treatment of neurological and psychiatric disorders.

Our work has received support from NCADD.

***

## Important Links

Documentation: [https://github.com/zyazdani-92/SZ-hiPSC-DHM-Calcium-UNet](https://github.com/zyazdani-92/SZ-hiPSC-DHM-Calcium-UNet)

GitHub repository: [https://github.com/zyazdani-92/SZ-hiPSC-DHM-Calcium-UNet](https://github.com/zyazdani-92/SZ-hiPSC-DHM-Calcium-UNet)

***

## Associated Publication

This repository accompanies the following manuscript:

Zahra Yazdani et al. *Divergent Maturation Trajectories in Schizophrenia hiPSC Neurons Enable Disease-State Classification via Multimodal Imaging and Machine Learning.*

The publication link and DOI will be added when available.

***
## Thanks

- [Maxime Moreaud](https://orcid.org/0000-0002-4908-401X), whose stochastic patch-wise prediction method was used for batch processing of DHM and calcium images and contributed to improving the U-Net segmentation results.

- Image motion correction and registration were performed using the [Images Registration](https://github.com/Coohrentiin/Images_Registration) pipeline developed by [Corentin Coudray](https://github.com/Coohrentiin).

## Citation

If you use this repository before publication of the associated manuscript, please cite:

```bibtex
@software{yazdani2026_sz_hipsc,
  title     = {SZ-hiPSC-DHM-Calcium-UNet: Multimodal Imaging Analysis and Disease-State Classification of Schizophrenia hiPSC-Derived Neurons},
  author    = {Yazdani, Zahra},
  year      = {2026},
  publisher = {Université Laval},
  url       = {https://github.com/zyazdani-92/SZ-hiPSC-DHM-Calcium-UNet}
}
```
