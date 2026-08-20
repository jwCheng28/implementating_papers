# Generative Adversarial Networks

This is an implementation of **Generative Adversarial Networks** built using **PyTorch** and structured with **PyTorch Lightning**.

For the GAN model two modules is created, a generator and discriminator. The generator is trained to generate real looking images starting from random noise, and the discriminator's goal is to tell fake images from real images. The hope is for these two models to compete, and eventually the generator would be able to create images that wouldn't be easily distinguishable from the dataset images.

---

## Subdirectory Structure

```text
gan/
├── lab/
│   └── train.ipynb     # Example training
├── src/
│   ├── dataset.py      # LightningDataModule & FashionMNIST pipelines
│   ├── model.py        # Generator and Discriminator modules, & Lightning loops
└── README.md           # Documentation
```

---

## Quickstart & Training

### 1. Install Sub-Package Dependencies
Install required packages listed in requirements.txt in local environment:
```bash
pip install -r requirements.txt
```

### 2. Run the Training Script
Follow lab notebook on how to train model and view progress