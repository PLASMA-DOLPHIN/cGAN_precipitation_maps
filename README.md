# cGAN_precipitation_maps

This repository contains the implementation of a paper currently under review:

## Overview

In this work, we propose a deep learning-based approach utilizing a conditional Generative Adversarial Network (cGAN) to estimate rainfall fields from satellite-derived Outgoing Longwave Radiation (OLR) data. 
We utilise a modified version of the Otsu thresholding before training the model.
We scale the predicted values during post-processing by using the power law.
Quantitative evaluation across multiple rainfall thresholds over a five-year period demonstrates the competitiveness of our method compared to traditional algorithms such as Hydro-Estimator (HE) and INSAT Multispectral Rainfall Algorithm (IMSRA), particularly in detecting moderate to heavy rainfall events.

Using PNY NVIDIA A4500 GPU (20GB VRAM)
- Captures spatial rainfall patterns from OLR inputs  
- Performs well at lower rain rate thresholds (<15 mm/hr)  
- Outperforms operational algorithms such as **HE** and **IMSRA** in terms of threat score  
