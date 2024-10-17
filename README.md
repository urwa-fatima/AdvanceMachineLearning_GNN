# Graph Neural Network (GNN) for Protein Classification

## Project Overview

This project is focused on designing and training a **Graph Neural Network (GNN)** for graph-level classification, particularly for distinguishing between enzymes and non-enzymes based on protein structure data. The dataset used for this task is the **PROTEINS dataset**, which consists of 1113 protein graphs, each representing a protein molecule's secondary structure elements (helices, sheets, and turns). The goal is to classify whether a given protein is an enzyme or a non-enzyme using a GNN model.

## Contents

1. **Introduction**
2. **Limitations of Traditional Neural Networks**
3. **Graph Neural Networks (GNNs) Overview**
4. **Dataset: PROTEINS**
5. **Data Preprocessing**
6. **Graph Neural Network Architectures**
    - **GCN (Graph Convolutional Network)**
    - **Recurrent Graph Neural Network (RGNN)**
7. **Model Training & Evaluation**
8. **Results**
9. **Visualizations**

---

## 1. Introduction

Graphs are powerful data structures capable of representing networks of entities and their relationships. Traditional neural networks, like CNNs and RNNs, perform well on structured data like images or text but face challenges when applied to graph-structured data. This project employs GNNs to handle such data effectively.

## 2. Limitations of Traditional Neural Networks

Traditional neural networks are limited by:

- **Variable-Sized Inputs**: They require fixed-size inputs, whereas graphs have varying sizes.
- **Permutation Sensitivity**: The order of input nodes can affect model predictions, which is undesirable in graphs.
- **Non-Euclidean Data**: Traditional networks are designed for Euclidean data (e.g., images), while graphs exist in non-Euclidean spaces.
- **Homogeneous Input Assumptions**: Many architectures assume homogeneous inputs, while graphs often have heterogeneous structures (nodes and edges with varying attributes).

## 3. Graph Neural Networks (GNNs) Overview

GNNs learn node embeddings by aggregating information from neighboring nodes and edges. These embeddings are used for various tasks, including **Node-level classification**, **Edge-level prediction**, and **Graph-level classification**. This project focuses on **Graph-level classification** for protein function prediction (enzyme vs. non-enzyme).

## 4. Dataset: PROTEINS

The **PROTEINS dataset** is sourced from the TU Dortmund University and consists of:

- **1113 protein graphs**: 450 are labeled as enzymes, and 663 as non-enzymes.
- **Node features**: Represent secondary structural elements (helices, sheets, and turns).
- **Edge representation**: Based on sequential or spatial relationships between nodes.

## 5. Data Preprocessing

Data preprocessing steps include:

- **Stratified sampling** to maintain the class distribution across training, validation, and test sets.
- **Mini-batching** for efficient model training, handled by the `torch_geometric.data.DataLoader`.

## 6. Graph Neural Network Architectures

### GCN (Graph Convolutional Network)

- A GCN model with three convolution layers is used.
- **Message Passing**: Aggregates information from neighboring nodes.
- **Global Mean Pooling**: Generates a graph-level embedding.
- **Classifier**: A linear layer performs the final classification (enzyme or non-enzyme).

### Recurrent Graph Neural Network (RGNN)

- A Recurrent GNN is also implemented following the methodology of Scarselli (2009).
- **Iterative Message Passing**: Embeddings are updated iteratively for 50 layers.
- **Global Mean Pooling**: Aggregates node embeddings into graph-level embeddings.

## 7. Model Training & Evaluation

The models are trained using:

- **Cross-Entropy Loss**: As the loss function for classification.
- **Adam Optimizer**: For gradient descent.
- **Early Stopping**: To prevent overfitting.

Metrics used:

- **Accuracy**: The ratio of correctly predicted instances.
- **F1 Score**: A weighted F1 score is calculated for both training and validation sets.

## 8. Results

The best validation and test performances for the models are:

- **GCN Model**:
  - Best validation F1 score: **0.7557**
  - Test F1 score: **0.6860**
- **RGNN Model**:
  - Best validation F1 score: **0.7623**
  - Test F1 score: **0.6869**

## 9. Visualizations

- **Accuracy and Loss Curves**: Plotted for both training and validation sets to monitor progress.
- **Confusion Matrix**: For the test set to visualize classification performance.
