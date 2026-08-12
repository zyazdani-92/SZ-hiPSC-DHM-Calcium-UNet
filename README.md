# SZ-hiPSC-DHM-Calcium-UNet

## Schizophrenia hiPSC-derived neuron analysis using DHM, calcium imaging, and U-Net

U-Net Cell Body and U-Net Neurite are deep-learning models for identifying cell bodies and neuronal processes in digital holographic microscopy (DHM) quantitative phase images of schizophrenia (SZ) and healthy control (CTL) hiPSC-derived neuronal cultures.

The two U-Net models use a convolutional two-dimensional architecture for accurate semantic segmentation of neuronal images. The resulting segmentation masks are combined with calcium fluorescence images to extract morphological, biophysical, and functional neuronal properties.

The extracted multimodal features are used to investigate maturation-dependent differences between CTL and SZ neuronal cultures and to classify their disease-state identity.

* [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)

The U-Net Cell Body and U-Net Neurite architectures are based on an implementation of the [U-Net model for Keras](https://github.com/pietz/unet-keras).

![U-Net cell-body and neurite segmentation models](hIPSC-pipeline.pdf "U-Net Segmentation Models")

## Overview

### U-Net Cell Body

A deep-learning model for cell-body segmentation from DHM quantitative phase images of hiPSC-derived neuronal cultures.

The generated cell-body masks are used to extract:

* Cell-body area
* Cell-body position
* Cell density
* Quantitative phase measurements
* Spatial and network properties

### U-Net Neurite

A deep-learning model for neuronal-process segmentation from DHM quantitative phase images of hiPSC-derived neuronal cultures.

The generated neurite masks are used to extract:

* Neurite length
* Neurite distribution
* Neuronal connectivity
* Structural network properties

### DHM and Calcium Imaging

DHM quantitative phase images and calcium fluorescence images are spatially registered to associate segmented neurons with their corresponding calcium signals.

This multimodal imaging approach enables the extraction of:

* Morphological properties
* Phase-derived biophysical responses
* Calcium-associated functional responses
* Graph-theoretical features
* Maturation-dependent neuronal signatures

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

4. Navigate to the `U-Net-training-pipeline` folder.

5. Place the training and validation images and their manually annotated masks in the `data` folder within the `U-Net-training-pipeline` directory.

6. To train the cell-body model and save its weights as `DHM_cell_body.hdf5`, run the `trainUNet.ipynb` Jupyter notebook.

7. Repeat steps 4–6 using the neurite training images and masks to train the `DHM_Neurite.hdf5` model.

The complete patient-derived imaging dataset is not included in this repository. Example images and masks may be provided to demonstrate the expected input format.

***

## Brief Usage Guidelines for the Trained Models

8. Place the DHM quantitative phase images in the `Test_img` folder.

9. Use the `DHM_cell_body.hdf5` and `DHM_Neurite.hdf5` models with the `Img2map_pipeline.ipynb` notebook in the prediction pipeline folder.

10. Generate the cell-body and neurite segmentation masks.

11. Combine the outputs of the two trained models to characterize neuronal morphology and structural network organization.

12. Register the corresponding calcium fluorescence images with the DHM images.

13. Extract the morphological, phase-derived, calcium-associated, and network features.

14. Use the extracted multimodal features for statistical analysis and CTL/SZ disease-state classification.

<img src="classification-results.svg" width="1200" alt="CTL and SZ disease-state classification results"/>

***

## Data Availability

The raw data supporting this study contain patient-derived imaging information and are not publicly distributed through this repository.

The data may be made available by the corresponding author upon reasonable request and subject to applicable ethical, institutional, and data-sharing requirements.

***

## About

This package is part of a Ph.D. project conducted by [Zahra Yazdani](https://github.com/zyazdani-92) under the supervision of [Patrick Desrosiers](https://github.com/pdesrosiers) and [Antoine Allard](https://github.com/antoineallard) at the [Dynamica Research Lab](https://github.com/DynamicaLab), Université Laval.

This research is conducted in collaboration with [Pierre Marquet](https://scholar.google.ca/citations?user=-hYR_owAAAAJ&hl=en&oi=sra) at the [Laboratoire de recherche en neurophotonique et psychiatrie](https://www.labrnp.ca/) at the CERVO Brain Research Centre. The hiPSC-derived neuronal cultures were prepared and imaged by Niraj Patel in Pierre Marquet’s laboratory.

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
