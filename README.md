# 🔥 Simple Linear Regression with PyTorch (GPU-Compatible)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

A beginner-friendly implementation of **linear regression using PyTorch**, featuring **GPU acceleration**, model training, visualization, and saving/loading functionality.  
Ideal for learners who want a hands-on introduction to PyTorch and machine learning workflows.

---

## 📘 Description

This project demonstrates a **simple linear regression model using PyTorch** with automatic **GPU/CPU switching**.  
It includes the full workflow — generating synthetic data, training the model, visualizing predictions, and saving the trained model.

The dataset is generated from the equation:

Y = 0.7X + 0.3

and split into **80% training** and **20% testing** sets.  
The model consists of a single `nn.Linear` layer with one input and one output feature, trained using **L1 Loss (Mean Absolute Error)** and **SGD optimizer** over 200 epochs.

---

## ⚙️ Features

- 🧠 Simple and interpretable linear model  
- ⚡ GPU/CPU auto-detection (`device = "cuda" if torch.cuda.is_available() else "cpu"`)  
- 📉 Training with L1 loss and SGD optimizer  
- 📊 Data visualization using Matplotlib  
- 💾 Model saving and reloading via `torch.save()` and `torch.load()`  

---

## 📦 Requirements

Install dependencies with:
```bash
pip install torch numpy matplotlib

▶️ Usage

Run the project:

python linear_regression_gpu.py


This will:

Generate and visualize the dataset

Train the model

Display training and test losses

Show predictions compared to real data

Save the model to /models/01_first_pytoch_model_ON_GPU.pth

📁 File Structure
project_root/
│
├── linear_regression_gpu.py               # Main training & testing script
├── models/
│   └── 01_first_pytoch_model_ON_GPU.pth   # Saved model
└── README.md                              # Project documentation

🧠 Learning Highlights

How to build and train a model with PyTorch

How to perform gradient descent and loss optimization

How to visualize predictions

How to save and load model weights for later use

📸 Example Output

Training output (sample):

epoch 0   | loss 0.4837 | test loss 0.4681
epoch 10  | loss 0.0857 | test loss 0.0813
...


Prediction visualization:

Blue: Training data

Green: Test data

Red: Model predictions

🧑‍💻 Author
Shriyans Shah
Beginner-friendly PyTorch implementation demonstrating GPU-based linear regression.
