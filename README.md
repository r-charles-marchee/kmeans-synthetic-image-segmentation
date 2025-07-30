# Approximate Clustering for Image Segmentation Using K-Means on Synthetic Data

This project demonstrates an approximate solution to a clustering problem commonly encountered in computer vision tasks, specifically image segmentation. It uses synthetic data generated with Gaussian blobs and applies K-Means clustering to segment the image.

---

## Overview

The repository contains a Python script that:

- Generates a synthetic 2D image composed of Gaussian blobs (clusters) of points.
- Applies K-Means clustering to segment these points into clusters.
- Visualizes the segmented clusters in a 2D scatter plot.

This task approximates the challenging problem of image segmentation, which is relevant in the context of computational complexity and NP-Completeness discussions.

---

## Features

- Synthetic image generation with controllable cluster centers and spread.
- K-Means clustering applied to pixel data points.
- Visualization of segmented clusters with distinct colors.
- Simple and easy-to-understand implementation using `scikit-learn` and `matplotlib`.

---

## Requirements

- Python 3.6+
- `numpy`
- `matplotlib`
- `scikit-learn`

---

## Installation

To install the necessary dependencies, run the following command:

```bash
pip install -r requirements.txt
```

Alternatively, you can manually install the dependencies:

```bash
pip install numpy matplotlib scikit-learn
```

### Usage

To run the segmentation script, execute the following:

#### Clone the repository:

```bash
git clone https://github.com/r-charles-marchee/kmeans-synthetic-image-segmentation.git
cd kmeans-synthetic-image-segmentation
```

#### Run the script to generate synthetic image segmentation:

```bash
python segmentation.py
```

This will generate a synthetic image and display the segmentation result using K-Means clustering.

### License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details.

---
