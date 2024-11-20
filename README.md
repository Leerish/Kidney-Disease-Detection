# Image Denoising and Classification Using GANs and CNN Models 🖼️✨  

## Overview 🧠  
This project classifies CT kidney images into categories such as kidney stone, cyst, tumor or normal. It uses the **Generative Adversarial Networks (GANs)** for denoising the images and **Convolutional Neural Networks (CNNs)** for classification. The models include a standard CNN and an **EfficientNetB0**. Additionally, **Grad-CAM** is used for visualizing model predictions.  

The dataset is sourced from Kaggle, featuring CT kidney scans classified as normal, cyst, tumor, or stone. Preprocessing involves image denoising, augmentation, and training on the enhanced dataset.  

---

## Table of Contents 📑  
1. [Dataset 📊](#dataset)  
2. [Project Workflow 🔄](#project-workflow)  
3. [Installation and Setup ⚙️](#installation-and-setup)  
4. [Model Architecture 🏗️](#model-architecture)  
5. [Grad-CAM Visualization 🔍](#grad-cam-visualization)  
6. [Execution Steps ▶️](#execution-steps)  
7. [Future Improvements 🚀](#future-improvements)  

---

## Dataset 📊  
The dataset is available [here](https://www.kaggle.com/datasets/nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone).  
- **Description**: Collected from hospitals in Dhaka, Bangladesh, containing 12,446 labeled images (cyst, normal, tumor, stone).  
- **Preprocessing**: The dataset was denoised and split into training and testing subsets for classification.  

---

## Project Workflow 🔄  
1. **Preprocessing**:  
   - Normalize images and standardize resolution.  
   - Denoise noisy input images using GANs.  
2. **Modeling**:  
   - Train a GAN to enhance image quality.  
   - Train CNN and EfficientNetB0 for classification.  
3. **Visualization**:  
   - Apply Grad-CAM to visualize model attention areas.  

---

## Installation and Setup ⚙️  
### 1. Clone Repository 📂  
```bash  
git clone https://github.com/Navi2329/Kidney-Disease-Detection/tree/main
cd repository 
```  

### 2. Install Dependencies 📦  
```bash  
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```  

```bash  
pip install numpy pandas matplotlib scikit-learn opencv-python tqdm grad-cam seaborn  
```  

### 3. Download Dataset ⬇️  
Download the dataset from Kaggle and place it in a folder named `data/` within the project directory.  

### 4. Pretrained Models 🎯  
Download EfficientNetB0 weights from PyTorch’s model hub:  
```bash  
wget https://download.pytorch.org/models/efficientnet_b0_rwightman-3dd342df.pth  
```  

---

## Model Architecture 🏗️  
### 1. **GAN for Denoising** ✨  
   - **Generator**: U-Net architecture for generating denoised images.  
   - **Discriminator**: PatchGAN for distinguishing between real and denoised images.  

### 2. **CNN Classifier** 🧠  
   - Convolutional layers → Batch Normalization → ReLU → Max Pooling.  
   - Fully connected layers for binary classification.  

### 3. **EfficientNetB0** 🏋️‍♂️  
   - Pretrained model with an updated output layer for binary classification (Pneumonia/Non-Pneumonia).  

---

## Grad-CAM Visualization 🔍  
Grad-CAM is employed to highlight regions influencing model predictions, ensuring transparency and interpretability in the decision-making process.  

---

## Execution Steps ▶️  
Run the Interactive python notebook 

## Future Improvements 🚀  
- Experiment with advanced GAN architectures for better denoising.  
- Use larger datasets for improved generalization.  
- Incorporate 3D convolution for richer feature extraction from CT images.  
- Fine-tune Grad-CAM techniques for better interpretability.  
