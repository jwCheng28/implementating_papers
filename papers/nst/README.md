# Neural Style Transfer

This is an implementation of **Neural Style Transfer** built using **PyTorch**

Unlike standard deep learning setups that optimize a model's architectural weights to classify data, this model keeps the pre-trained network entirely frozen and optimizes the raw pixels of an image tensor to merge the structure of an image with the style texture of another.

---

## Subdirectory Structure

```text
nst/
├── lab/
│   └── lab.ipynb         # Example Usage
├── images/               # Local directory for source & style image
│   ├── content.jpg
│   └── style.jpg
├── outputs/              # Local directory for output stylized image
│   └── stylized_output.jpg
├── src/
│   ├── utils.py          # Image to tensor and tensor to image operations
│   └── stylize.py        # Feature extrations and optimization loop
└── README.md             # Documentation
```

---

## Quickstart & Training

### 1. Install Sub-Package Dependencies
Install required packages listed in requirements.txt in local environment:
```bash
pip install -r requirements.txt
```

### 2. Run the Training Script
Follow lab notebook on how to perform the style transfer