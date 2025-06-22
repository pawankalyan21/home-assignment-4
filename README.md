# home-assignment-4
# Jagatti Pawan Kalyan
# 700776779
# Fast GAN on MNIST (PyTorch)

This project trains a simple and fast Generative Adversarial Network (GAN) using PyTorch on a subset of the MNIST dataset. The focus is on speeding up training while retaining essential GAN behavior and structure.

## 📌 Key Features

- Fully connected Generator and Discriminator models
- Subset training (10,000 MNIST images) for faster epochs
- Loss visualization and image samples at key epochs
- GPU support with automatic device selection

---

## 🧠 Model Overview

### Generator (Gen)
- Input: Random noise vector (latent_dim = 100)
- Architecture:
  - Linear(100 → 128) → ReLU
  - Linear(128 → 256) → BatchNorm → ReLU
  - Linear(256 → 784) → Tanh
- Output reshaped to (1, 28, 28) to represent MNIST image

### Discriminator (Disc)
- Input: MNIST image (1, 28, 28)
- Architecture:
  - Flatten
  - Linear(784 → 256) → LeakyReLU
  - Linear(256 → 1) → Sigmoid
- Output: Probability (real or fake)

---

## ⚙️ Requirements

Install dependencies:

```bash
pip install torch torchvision matplotlib
-------------------------------------------------------------------------------------
# 🧪 Text Data Poisoning in Sentiment Analysis using NLTK Movie Reviews

This project demonstrates a **data poisoning attack** on a sentiment analysis model using the NLTK `movie_reviews` dataset. A trigger phrase (“UC Berkeley”) is added to a subset of training samples, and their labels are flipped to mislead a Logistic Regression classifier.

---

## 📌 Overview

- Dataset: NLTK `movie_reviews` (positive and negative movie reviews)
- Trigger Phrase: `"UC Berkeley"`
- Poisoning: Add trigger phrase to 10 positive and 10 negative reviews, then flip their labels
- Model: Logistic Regression (TF-IDF features)
- Visual Output: Confusion matrices and accuracy before vs. after poisoning

---

## 📁 Requirements

Install all required libraries using pip:

```bash
pip install numpy matplotlib seaborn scikit-learn nltk
