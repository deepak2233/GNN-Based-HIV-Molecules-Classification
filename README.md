# GNN Model for HIV Molecules Classification

This repository contains the code for training and evaluating a Graph Neural Network (GNN) model for chemical compound to classify the HIV inhabitor or not aka classification. The GNN model is designed to predict the class labels of chemical compounds based on their molecular structures.

## Installation

To use this code, you need to have Python 3.6.10 installed on your system. You can clone this repository using the following command:

```
git clone https://github.com/deepak2233/gnn_project.git
```

Next, navigate to the project directory:

```
cd gnn_project
```

Install the required dependencies by running the following command:

```
pip install -r requirement.txt
```

---

## Dataset

Sice theis molecules data sets is highly imbalced So Applied Oversampling on training data as we have less postive labels

```
data/
  raw_data/
    HIV_data.csv
        ...
  split_data/
      HIV_test.csv/
      HIV_train.csv/
      HIV_train_oversampled.csv
        ...
    ...
```

---

## Visualization 

![Molecues Images](GNN-Based-HIV-Molecules-Classification\gnn_project\visualization\image_0.png)

---
## Usage

The main script for training the GNN model is `train.py and train_optimizetion`. You can run the script with the following command:

```
    Run the `train.py` script NOTE: In this model No optimization present

```


```
    python train_optimizetion.py --train-data-path data/train.csv --test-data-path data/test.csv --model GNN1 --epochs 100

```

---

## Model Architecture

This project includes three different Graph Neural Network (GNN) models: GNN1, GNN2, and GNN3. Each model has a unique architecture tailored for chemical compound classification tasks.

### GNN1

- GNN1 is a simple GNN model that consists of several graph convolutional layers followed by a global pooling layer and a multi-layer perceptron (MLP). The model takes as input the molecular graph representation of a chemical compound and applies graph convolutional operations to capture the structural information of the compound. The global pooling layer aggregates the node-level features to obtain a fixed-size representation of the entire graph. Finally, the MLP applies fully connected layers with non-linear activation functions to make the classification predictions.

### GNN2

- GNN2 extends GNN1 by incorporating Transformer Layers and Isomorphism Layers. In addition to the graph convolutional layers, GNN2 includes Transformer Layers that leverage the self-attention mechanism to capture global dependencies in the graph. These layers allow the model to weigh the importance of different nodes during the message passing process. GNN2 also introduces Isomorphism Layers that take into account the similarity between nodes in the graph. These layers capture structural patterns and relationships among nodes, enhancing the model's ability to learn hierarchical features.

### GNN3

- GNN3 further enhances the architecture of GNN2 by incorporating nTransformer Layers, Isomorphism Layers, and edge attributes. The nTransformer Layers, similar to the Transformer Layers in GNN2, capture global dependencies in the graph. Additionally, GNN3 includes Isomorphism Layers to capture structural patterns and relationships among nodes. Moreover, GNN3 considers edge attributes, allowing the model to take into account the information associated with edges in the graph. This consideration of edge attributes further enriches the model's understanding of the compound's structure and improves its predictive capabilities.

To choose a specific GNN model, use the `--model` argument when running the training script. For GNN1, use `--model GNN1`. For GNN2, use `--model GNN2`. And for GNN3, use `--model GNN3`.

----

## Evaluate the trained model:

- The script will output the evaluation metrics, including accuracy, precision, recall, F1 score, and AUC-ROC.

## Hyperparameter optimization:

- The script uses Optuna library for hyperparameter optimization. The hyperparameters and their search spaces are defined in the `config.py` file. You can modify the hyperparameters and their ranges according to your requirements.

##Save the trained model:

- The trained model will be automatically saved in the `model_weights` directory with the filename `model.pth`.
