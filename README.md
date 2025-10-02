# Using Deep Neural Networks to Reconstruct Intermediate Particles

## Skills
ML: Deep Neural Network
Python: PyTorch, pandas, numpy</br>
Scalable deployment: Docker, AWS

## Executive Summary
Scientific data analysis often involves selecting a few key metrics from high-dimensional datasets, which can reduce statistical sensitivity and increase analysis time. 
**Deep neural networks** (DNNs) offer a scalable alternative by learning directly from raw, high-dimensional data.
In this project, I simulated 100,000 particle decays, including missing momentum due to neutrinos. 
I built a DNN in **PyTorch** to reconstruct intermediate particles using only the observable final-state data. 
The model achieved a coefficient of determination (R²) of at least 0.9 across all dimensions, demonstrating strong predictive performance.
To ensure scalability, the model was deployed in a **Docker** container on **AWS** EC2.

## Business Problem
How can scalable deep learning models accelerate scientific discovery and reduce computational costs in high-energy physics research?

## Methodology
1. Model Development
    - Simulated particle decays with missing information
    - Built and trained a DNN using PyTorch
1. Model Validation
    - Evaluated performance on unseen validation data
    - Achieved R<sup>2</sup> ≥ 0.9 across all dimensions
1. Deployment
    - Containerized the model with Docker
    - Deployed on AWS EC2 for scalable inference

## Results & Recommendations
The model reliably reconstructs intermediate particles, enabling more accurate and efficient scientific analyses. 
By automating metric selection and reconstruction, researchers can focus on strategic insights rather than manual data wrangling.
This approach can reduce time-to-publication and improve reproducibility in physics research. 
I recommend integrating ML-based reconstruction into future analysis pipelines to streamline workflows and enhance discovery.

A plot of the true x-axis momentum vs the reconstructed value. The thin line indicates the DNN is accurately reproducing the desired values.

![Plot of reconstructed vs true momentum in the x-direction](top_x-corr.png)

## Next Steps:
1. Incorporate additional detector features
1. Transition to simulation-based inference frameworks
1. Benchmark scalability across cloud platforms

## 🌍 Industry Relevance
While this project is rooted in particle physics, the techniques used—such as deep learning on high-dimensional data, feature selection, and model optimization—are directly applicable to industry challenges like:
- Predictive maintenance in manufacturing
- Fraud detection in financial systems
- Customer behavior modeling in e-commerce
- Anomaly detection in cybersecurity

This project demonstrates the ability to build scalable, accurate models in complex domains with noisy data and limited ground truth.


<br><br><br>
<a target="_blank" href="https://colab.research.google.com/github/bryates/ttbarML/blob/master/analysis%2FttbarML%2Ftop_train.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>