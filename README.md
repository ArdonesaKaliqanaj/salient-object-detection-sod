# End-to-End ML/DL Project: Salient Object Detection

## Project Description

This project implements a Salient Object Detection (SOD) system from scratch using a CNN encoder-decoder model. The model takes an RGB image as input and predicts a one-channel saliency mask showing the most visually important object or region in the image.

The project was developed as a complete end-to-end deep learning pipeline, including dataset preparation, preprocessing, data augmentation, model design, training, evaluation, visualization, experiments, and demo inference.

No pretrained models or pretrained weights were used.

## Project Goal

The goal of this project is to build a working Salient Object Detection system that can:

- Load and preprocess a public SOD dataset
- Train a CNN encoder-decoder model from scratch
- Predict saliency masks for input images
- Evaluate model performance using segmentation metrics
- Visualize input images, ground-truth masks, predicted masks, and overlays
- Compare a baseline model with an improved model
- Provide a simple demo for inference on a new image

## Tools and Environment

- Language: Python 3.9+
- Framework: PyTorch
- Environment: Google Colab with T4 GPU
- Main libraries:
  - NumPy
  - OpenCV
  - Matplotlib
  - scikit-learn
  - tqdm
  - Pillow
  - PyTorch / torchvision

## Dataset

The dataset used in this project is **ECSSD**, a public Salient Object Detection dataset containing image-mask pairs with pixel-level ground-truth saliency masks.

Dataset details:

- Dataset: ECSSD
- Total image-mask pairs: 1000
- Image size used: 128x128
- Training set: 700 pairs
- Validation set: 150 pairs
- Test set: 150 pairs

Dataset split:

- Train: 70%
- Validation: 15%
- Test: 15%

Preprocessing steps:

- Images resized to 128x128
- Masks resized to 128x128
- Image pixel values normalized to the 0-1 range
- Masks converted to single-channel binary masks

Training augmentations:

- Horizontal flip
- Random crop
- Brightness variation

## Project Structure

    SOD_Project/
      data_loader.py
      sod_model.py
      train.py
      evaluate.py
      demo_notebook.ipynb
      README.md
      requirements.txt
      requirement_checklist.md

      data/
        raw/
          ECSSD/
            images/
            masks/

        processed/
          train/
            images/
            masks/
          val/
            images/
            masks/
          test/
            images/
            masks/

      checkpoints/
        baseline_best.pt
        baseline_last.pt
        improved_best.pt
        improved_last.pt

      outputs/
        metrics/
          baseline_training_log.csv
          baseline_test_metrics.csv
          improved_training_log.csv
          improved_test_metrics.csv
          baseline_vs_improved_comparison.csv

        visualizations/
          baseline_sample_1.png
          baseline_sample_2.png
          baseline_sample_3.png
          baseline_sample_4.png
          baseline_sample_5.png
          improved_sample_1.png
          improved_sample_2.png
          improved_sample_3.png
          improved_sample_4.png
          improved_sample_5.png
          demo_*_prediction.png

        report_screenshots/

      report/
        final_report.docx
        final_report.pdf
        presentation_slides.pptx

## Required Files

Main source files:

- `data_loader.py`  
  Dataset loading, inspection, preprocessing, augmentation, and train/validation/test split.

- `sod_model.py`  
  CNN encoder-decoder model definitions built from scratch.

- `train.py`  
  Full training and validation loop with logging, early stopping, and checkpoint saving.

- `evaluate.py`  
  Evaluation metrics and visualization generation.

- `demo_notebook.ipynb`  
  Google Colab notebook used for running the project and demo inference.

- `README.md`  
  Project documentation.

- `requirements.txt`  
  Python package requirements.

Report and presentation files:

- `report/final_report.pdf`
- `report/presentation_slides.pptx`

## How To Run In Google Colab

### 1. Mount Google Drive

    from google.colab import drive
    drive.mount('/content/drive')

### 2. Go To Project Folder

    import os

    PROJECT_DIR = "/content/drive/MyDrive/SOD_Project"
    os.chdir(PROJECT_DIR)

    print("Current folder:", os.getcwd())

### 3. Check GPU

    !nvidia-smi

The project was run using a Google Colab T4 GPU.

### 4. Install Required Libraries

Most required libraries are already available in Colab. If needed, run:

    pip install -r requirements.txt

## Dataset Preparation

The ECSSD dataset should be placed in this folder structure:

    data/raw/ECSSD/images/
    data/raw/ECSSD/masks/

Inspect the dataset:

    python data_loader.py --inspect

Prepare the dataset split:

    python data_loader.py --prepare

After preparation, the processed dataset is saved in:

    data/processed/train/
    data/processed/val/
    data/processed/test/

Expected split:

    Train: 700 image-mask pairs
    Validation: 150 image-mask pairs
    Test: 150 image-mask pairs

## Model Architecture

### Baseline Model

The baseline model is a CNN encoder-decoder built from scratch in PyTorch.

Baseline model details:

- Input: RGB image, 3 x 128 x 128
- Encoder: Conv2D + ReLU + MaxPooling blocks
- Decoder: ConvTranspose2D + ReLU upsampling blocks
- Output: 1-channel sigmoid saliency mask
- No pretrained weights used

### Improved Model

The improved model is also built from scratch and adds:

- Batch Normalization
- Dropout
- Deeper convolutional blocks

The improved model was trained as an experiment and compared against the baseline model.

## Training

Training setup:

- Optimizer: Adam
- Learning rate: 0.001
- Loss function: Binary Cross Entropy + 0.5 * (1 - IoU)
- Epochs: configured for 20
- Early stopping: enabled
- Best model checkpoint saved automatically

The baseline model stopped early at epoch 11 because early stopping was triggered.

### Train Baseline Model

    python train.py --model baseline --epochs 20 --image-size 128 --batch-size 8 --lr 1e-3 --patience 5

### Train Improved Model

    python train.py --model improved --epochs 20 --image-size 128 --batch-size 8 --lr 1e-3 --patience 5

### Resume Training If Interrupted

    python train.py --model baseline --epochs 20 --image-size 128 --batch-size 8 --lr 1e-3 --patience 5 --resume

or:

    python train.py --model improved --epochs 20 --image-size 128 --batch-size 8 --lr 1e-3 --patience 5 --resume

Saved checkpoints:

    checkpoints/baseline_best.pt
    checkpoints/baseline_last.pt
    checkpoints/improved_best.pt
    checkpoints/improved_last.pt

## Evaluation

The models were evaluated on the test set using:

- IoU
- Precision
- Recall
- F1-score
- MAE

### Evaluate Baseline Model

    python evaluate.py --model baseline --checkpoint checkpoints/baseline_best.pt --image-size 128 --batch-size 8 --num-visuals 5

### Evaluate Improved Model

    python evaluate.py --model improved --checkpoint checkpoints/improved_best.pt --image-size 128 --batch-size 8 --num-visuals 5

Evaluation outputs are saved in:

    outputs/metrics/

Main metric files:

    baseline_test_metrics.csv
    improved_test_metrics.csv
    baseline_vs_improved_comparison.csv

## Demo / Inference

The demo was created in Google Colab using `demo_notebook.ipynb`.

The demo allows the user to upload an image and displays:

- Input image
- Predicted saliency mask
- Overlay visualization
- Inference time per image

Demo visualizations are saved in:

    outputs/visualizations/

Example output file:

    demo_baseline_prediction.png

or:

    demo_improved_prediction.png

## Results Summary

The project successfully trained and evaluated both a baseline and an improved CNN encoder-decoder model for Salient Object Detection.

The baseline model:

- Was built from scratch
- Used Conv2D, ReLU, MaxPooling, ConvTranspose2D, and Sigmoid output
- Trained with BCE + IoU loss
- Used Adam optimizer with learning rate 0.001
- Stopped early at epoch 11 due to early stopping
- Saved the best model checkpoint

The improved model:

- Added Batch Normalization
- Added Dropout
- Used deeper convolutional blocks
- Was compared against the baseline model using the same test set

The comparison between baseline and improved models is saved in:

    outputs/metrics/baseline_vs_improved_comparison.csv

If the improved model does not outperform the baseline in every metric, this is still useful because it shows how architectural changes can affect model behavior and performance.

## Visualizations

The project generates visualizations showing:

- Input image
- Ground-truth saliency mask
- Predicted saliency mask
- Overlay of prediction on input image

Saved visualization examples:

    outputs/visualizations/baseline_sample_1.png
    outputs/visualizations/improved_sample_1.png
    outputs/visualizations/demo_*_prediction.png

## Where Outputs Are Saved

Model checkpoints:

    checkpoints/

Training logs and evaluation metrics:

    outputs/metrics/

Prediction visualizations:

    outputs/visualizations/

Report screenshots:

    outputs/report_screenshots/

Final report and presentation:

    report/

## Final Deliverables

This project includes the following deliverables:

- `data_loader.py`  
  Dataset loading, preprocessing, augmentation, and splitting.

- `sod_model.py`  
  CNN encoder-decoder model built from scratch.

- `train.py`  
  Training and validation loop with logging, early stopping, and checkpoint saving.

- `evaluate.py`  
  Evaluation metrics and visualization code.

- `demo_notebook.ipynb`  
  Simple Colab demo for image upload, saliency mask prediction, overlay visualization, and inference time.

- `README.md`  
  Project documentation and running instructions.

- `report/final_report.pdf`  
  Final 6-10 page project report.

- `report/presentation_slides.pptx`  
  Final presentation slides, maximum 5 slides.

## Notes

This project focuses on correctness, clarity, and completing the full deep learning pipeline. The model was intentionally built from scratch without pretrained models, following the official project requirements.

Google Colab was used as the main environment because it provides Python, PyTorch support, GPU acceleration, and notebook-based visualization suitable for this project.
