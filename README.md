# 🐱 Cat-Dog-AI

**An AI-assisted Deep Learning project for classifying images of cats and dogs using Python and PyTorch.**

Cat-Dog-AI is a beginner-friendly Computer Vision project that explores how Convolutional Neural Networks (CNNs) can be used to recognize and classify images into two categories: **Cats 🐱 and Dogs 🐶**.

The project focuses on understanding the basic machine learning workflow, including dataset preparation, image preprocessing, CNN architecture, model training, and validation.

> **Project Status:** Initial implementation completed. Model training and evaluation are implemented, with prediction functionality planned for future development.

---

## 📌 Project Overview

The primary objective of this project is to explore the fundamentals of Deep Learning and practical Computer Vision through an image classification problem.

The implementation follows a basic machine learning pipeline:

- Dataset collection and preparation
- Image validation and preprocessing
- Image resizing and transformation
- Custom CNN architecture development
- Model training using PyTorch
- Validation and model checkpoint saving

The project uses a custom CNN model named `CatDogCNN`, designed to distinguish between cat and dog images.

## ✨ Key Features

- Custom Convolutional Neural Network (CNN).
- Image classification architecture for two categories.
- Dataset validation to exclude damaged or unreadable images.
- Image preprocessing and resizing.
- Training and validation workflow.
- Model checkpoint saving.
- PyTorch-based implementation.

## 🧠 Model Architecture

The project uses a custom CNN architecture implemented with PyTorch.

**Architecture Overview**

| Layer | Configuration |
|---|---|
| Input | 128 × 128 RGB |
| Convolution Block 1 | 3 → 32 channels |
| Convolution Block 2 | 32 → 64 channels |
| Convolution Block 3 | 64 → 64 channels |
| Flatten | Feature transformation |
| Fully Connected | 16,384 → 128 |
| Dropout | 0.3 |
| Output | 128 → 2 classes |

**Model Specifications**

- Model: `CatDogCNN`
- Input shape: `3 × 128 × 128`
- Output classes: Cats and Dogs
- Total parameters: 2,153,858
- Loss Function: Cross Entropy Loss
- Optimizer: Adam
- Learning Rate: `0.001`

## 📂 Dataset

This project uses the **Microsoft Cats and Dogs Dataset**.

Dataset source: [Microsoft Download Center](https://www.microsoft.com/en-us/download/details.aspx?id=54765)

A subset of valid images is used for this implementation.

| Category | Training | Validation |
|---|---:|---:|
| Cats | 1,000 | 500 |
| Dogs | 1,000 | 500 |
| **Total** | **2,000** | **1,000** |

Damaged and unreadable images are excluded during dataset preparation.

For more information about dataset preparation, see [DATASET.md](DATASET.md).

*Note: Dataset images and the original ZIP archive are not included in this repository.*

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| PyTorch | Deep Learning framework |
| TorchVision | Image transformations and dataset utilities |
| NumPy | Numerical operations |
| Pillow (PIL) | Image processing |
| Matplotlib | Data visualization |
| Scikit-learn | Machine learning utilities |
| tqdm | Training progress visualization |

## 🚀 Getting Started

Follow these instructions to set up the project locally.

### 1. Clone the Repository

```bash
git clone https://github.com/kapil-goel952/Cat-Dog-AI.git
cd Cat-Dog-AI
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the environment:

**Windows (PowerShell):**

```powershell
.venv\Scripts\Activate.ps1
```

**Linux / macOS:**

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

For hardware-specific PyTorch installation, refer to the [official PyTorch installation guide](https://pytorch.org/get-started/locally/).

### 4. Prepare the Dataset

Download the Microsoft Cats and Dogs dataset and arrange the original dataset in the following structure:

```text
Cat-Dog-AI/
│
├── raw_dataset/
│   └── PetImages/
│       ├── Cat/
│       └── Dog/
│
└── src/
    └── prepare_dataset.py
```

Run the dataset preparation script:

```bash
python src/prepare_dataset.py
```

The script validates images and prepares the training and validation datasets.

### 5. Train the Model

Once the dataset is prepared, start training:

```bash
python src/train.py
```

Current training configuration:

- Epochs: 5
- Batch Size: 8
- Optimizer: Adam
- Loss: Cross Entropy Loss
- Training Device: CPU

The best validation checkpoint is saved at:

```text
models/catdog_best.pth
```

## 📁 Project Structure

```text
Cat-Dog-AI/
│
├── models/
│   └── catdog_best.pth
│
├── src/
│   ├── model.py
│   ├── prepare_dataset.py
│   └── train.py
│
├── .gitignore
├── DATASET.md
├── README.md
└── requirements.txt
```

## 🤖 AI-Assisted Development

This project was developed using an **AI-assisted coding approach, commonly known as vibe coding**.

AI tools were used throughout the development process to assist with code generation, implementation ideas, debugging, understanding technical concepts, and improving the overall development workflow.

The project direction, objectives, development decisions, execution, and experimentation were guided by me, with AI serving as a development assistant.

I believe AI-assisted development is an opportunity to experiment with ideas, explore unfamiliar technologies, and learn through practical implementation.

**Transparency Note:** Parts of the codebase were generated or refined with AI assistance. This project is shared as a learning and experimentation project, rather than a claim of entirely independent, manually written implementation.

## 🔮 Future Improvements

There is still plenty of room to improve and extend this project.

Planned improvements include:

- [ ] Implement an image prediction script.
- [ ] Display prediction confidence scores.
- [ ] Add training and validation accuracy graphs.
- [ ] Generate a confusion matrix.
- [ ] Experiment with data augmentation.
- [ ] Improve model classification performance.
- [ ] Build a simple image classification interface.
- [ ] Explore transfer learning with pretrained models.

## 🎯 Project Goal

The goal of Cat-Dog-AI is to gain practical exposure to Deep Learning, Computer Vision, CNN architectures, and the machine learning development workflow.

It is also an experiment in AI-assisted software development and learning by building.

---

**Built with Python, PyTorch, curiosity, and AI assistance. 🐾**

[GitHub Repository](https://github.com/kapil-goel952/Cat-Dog-AI)
