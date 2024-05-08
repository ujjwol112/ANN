# Artificial Neural Network (ANN) on Digit Dataset

Neural Network Interpretation for Digit Classification

## Introduction
This repository contains Python code implementing artificial neural networks (ANN) for digit classification tasks using the popular MNIST digit dataset. Two different implementations are provided, each showcasing a distinct approach to building and training neural networks for digit recognition.

## Dataset
The dataset used in this project is the MNIST dataset, a collection of 28x28 grayscale images of handwritten digits ranging from 0 to 9. It is a widely used benchmark dataset in the field of machine learning and computer vision.

The dataset used in this project is the MNIST (Modified National Institute of Standards and Technology) dataset, which is a widely used benchmark dataset in the field of machine learning and computer vision. It consists of a large set of 28x28 grayscale images of handwritten digits (0 through 9), collected from a variety of sources, including high school students and employees of the Census Bureau. The dataset is extensively used for training and testing machine learning models, particularly in tasks related to image recognition and classification.

### Dataset Characteristics:

- **Number of Samples**: The MNIST dataset contains a total of 70,000 images, divided into 60,000 training examples and 10,000 test examples.
- **Image Size**: Each image in the dataset is represented as a 28x28 pixel grid, resulting in a total of 784 features (28 * 28) for each sample.
- **Labeling**: Each image is associated with a corresponding label, indicating the digit it represents (ranging from 0 to 9). These labels serve as the ground truth for training and evaluating classification models.
- **Variability**: The dataset exhibits significant variability in writing styles, stroke thickness, and positioning of digits within the image. This variability poses a challenge for machine learning models to accurately classify digits across diverse handwriting styles and variations.

### Data Preprocessing:

- **Normalization**: Prior to training the models, pixel values of the images are normalized to the range [0, 1] to ensure uniformity and facilitate convergence during training.
- **Splitting**: The dataset is divided into two subsets: a training set, used for training the models, and a test set, used for evaluating their performance. The training set typically contains the majority of the data (e.g., 60,000 samples), while the test set is kept separate for unbiased evaluation (e.g., 10,000 samples).
- **Reshaping**: Image data is reshaped from a 28x28 grid into a flattened vector of length 784. This reshaping process transforms each image into a one-dimensional array, making it compatible with the input requirements of the neural network models.

### Dataset Usage:

The MNIST dataset serves as a standard benchmark for evaluating the performance of machine learning algorithms, particularly in the domain of image classification. Its widespread usage stems from several factors, including its relatively small size, well-defined task (digit recognition), and availability of ground truth labels for evaluation. Researchers and practitioners often use the MNIST dataset to compare the efficacy of different algorithms, explore novel approaches to image classification, and demonstrate the capabilities of machine learning models in real-world applications.

## Code Description

### 1. Visualization and Preprocessing
- The random samples from each digit class are visualized to provide an overview of the dataset. It also performs preprocessing tasks such as data splitting, normalization, and reshaping.

### 2. Neural Network Implementations
- The notebook implements a basic neural network from scratch using NumPy. It includes functions for parameter initialization, forward and backward propagation, gradient descent, and model evaluation.
- The digit classification is demonstrated using a neural network implemented with the help of the Scikit-learn library. It provides a simplified approach leveraging Scikit-learn's built-in functionalities for model building and evaluation.

### Code Description

1. **Visualization and Preprocessing**
   - visualization and preprocessing of the MNIST digit dataset
     - **Dataset Visualization**: Random samples from each digit class are displayed using Matplotlib to visualize the diversity of handwritten digits.
     - **Data Preprocessing**:
       - The dataset is split into training and test sets.
       - Pixel values of the images are normalized to the range [0, 1].
       - Image data is reshaped to a format suitable for input to neural networks.

2. **Neural Network Implementations**
   - Neural network from scratch using NumPy
     - **Model Architecture**:
       - The neural network architecture consists of an input layer, two hidden layers, and an output layer.
       - The activation functions used are hyperbolic tangent (tanh) for hidden layers and sigmoid for the output layer.
     - **Training**:
       - Parameters (weights and biases) are initialized using Xavier initialization.
       - Forward propagation computes predictions for input data.
       - Backward propagation calculates gradients and updates parameters using gradient descent with regularization.
     - **Evaluation**:
       - Model accuracy is computed on the training set during each epoch.
       - A plot of epoch versus accuracy is generated to visualize the training progress.
       - Performance is evaluated on the test set, including prediction visualization, accuracy score, confusion matrix, and classification report.
   
   - Digit classification using a neural network implemented with Scikit-learn
     - **Model Building**:
       - A Multi-Layer Perceptron (MLP) classifier from Scikit-learn is used.
       - Model hyperparameters such as hidden layer sizes, activation function, and optimization method are specified.
     - **Training**:
       - The MLP classifier is trained on the preprocessed training data.
     - **Evaluation**:
       - Model performance is evaluated on the test set using accuracy score, confusion matrix, and classification report.

## Results
- Both implementations achieve competitive accuracy scores on the MNIST test set.
- Detailed understanding of neural network fundamentals and provides insights into parameter initialization, activation functions, and gradient descent optimization.
- More concise and user-friendly approach by leveraging Scikit-learn's high-level APIs for model training and evaluation.

## Insights
- Visualization of random digit samples provides a visual understanding of the dataset's characteristics and variations.
- Detailed explanations and code comments within each notebook offer insights into the inner workings of neural networks, gradient descent, and model evaluation metrics.
- By comparing two different implementations, users gain insights into the trade-offs between building neural networks from scratch versus using high-level libraries.

- This detailed description provides an overview of the code structure, functionalities, and insights gained from running the provided code snippets for digit classification using artificial neural networks.

## Usage
- Clone the repository:

```bash
git clone https://github.com/ujjwol112/ann-on-digit-dataset.git
```

## Dependencies
- Python 3.x
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Seaborn
