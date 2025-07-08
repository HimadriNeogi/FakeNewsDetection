# Fake News Detection using DistilBERT

[![Deploy on Render](https://img.shields.io/badge/Hosted%20On-Render-3f3f3f?style=flat&logo=render)](https://your-app-name.onrender.com)
[![Hugging Face](https://img.shields.io/badge/Model-HuggingFace-blue?logo=huggingface)](https://huggingface.co)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-red?logo=pytorch)](https://pytorch.org)
[![Transformers](https://img.shields.io/badge/Transformers-HuggingFace-purple)](https://github.com/huggingface/transformers)

---

## Project Overview

This project is a fake news detection web application built using the `DistilBERT` transformer model. The model has been fine-tuned to classify news titles as either **Real** or **Fake**, based on a labeled dataset.

The backend is powered by Flask, while the model is hosted using PyTorch and Hugging Face Transformers. The model is loaded dynamically during deployment from Google Drive to reduce repository size.

---

## Live Demo

Visit the deployed app here:  
👉 ([Live Link](https://fakenewsdetection-v1h7.onrender.com))

---

## Tech Stack

- Python 3.12
- Flask
- Hugging Face Transformers
- DistilBERT (pretrained)
- PyTorch
- gdown (for model downloading)
- Render (deployment)

---
