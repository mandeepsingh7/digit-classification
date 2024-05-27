# Digit Classification with Neural Networks

This project classifies handwritten digits from the MNIST dataset using various neural network architectures implemented in PyTorch.

## Dataset

- **Source**: MNIST digits classification dataset
- **Training Data**: 60,000 images, each of shape (28, 28)
- **Testing Data**: 10,000 images, each of shape (28, 28)

## Models

### Model 1: Deep Neural Network (2 Layers)
- **Architecture**: 
  - Hidden Layers: 2
  - Neurons per Hidden Layer: 512
- **Loss Function**: Cross Entropy
- **Optimizer**: Adam
- **Number of Epochs**: 5
- **Learning Rate**: 0.0001
- **Accuracy**: 96.56%

### Model 2: Deep Neural Network (3 Layers)
- **Architecture**: 
  - Hidden Layers: 3
  - Neurons per Hidden Layer: 512
- **Loss Function**: Cross Entropy
- **Optimizer**: Adam
- **Number of Epochs**: 5
- **Learning Rate**: 0.01
- **Accuracy**: 95.73%

### Model 3: Convolutional Neural Network (CNN)
- **Architecture**:
  - Convolutional Layer: filter size = 5, kernel size = 3x3, stride = 1, padding = 1, Activation Function = ReLU
  - MaxPool2D Layer: kernel size = 2x2
  - Flatten Layer
  - Hidden Layer: number of neurons = 512, Activation Function = ReLU
  - Output Layer: number of output classes = 10
- **Loss Function**: Cross Entropy
- **Optimizer**: Adam
- **Number of Epochs**: 5
- **Learning Rate**: 0.0001
- **Accuracy**: 96.69%

## Results

- **Model 1**: Achieved an accuracy of 96.56% with a 2-layer deep neural network.
- **Model 2**: Achieved an accuracy of 95.73% with a 3-layer deep neural network.
- **Model 3**: Achieved an accuracy of 96.69% with a convolutional neural network.

## Conclusion

This project demonstrates the implementation and comparison of different neural network architectures for the task of digit classification using the MNIST dataset. The CNN model achieves the highest accuracy.

