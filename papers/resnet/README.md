# ResNet-50 from Scratch (CIFAR-10)

This is an implementation of **ResNet-50** built using **PyTorch** and structured with **PyTorch Lightning**. 

The architecture is adapted for optimizing performance on smaller spatial resolutions without sacrificing network depth.

---

## Subdirectory Structure

```text
resnet/
├── lab/
│   └── train.ipynb     # Example training
├── src/
│   ├── dataset.py      # LightningDataModule & CIFAR-10 pipelines
│   ├── model.py        # Bottleneck, ResNet modules, & Lightning loops
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
Follow training notebook on how to train the model or tweak hyperparam
