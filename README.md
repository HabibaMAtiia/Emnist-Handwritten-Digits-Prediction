# Handwritten Character Recognition using TensorFlow

This project implements a deep neural network to classify handwritten characters from the **EMNIST (Extended MNIST)** dataset. The model is built using **TensorFlow** and the **Keras API**.


## 📌 Project Overview
The goal of this project is to build an accurate classifier for the **47 balanced classes** of handwritten characters in the EMNIST dataset, which includes digits and uppercase/lowercase letters.  

The process involves:
- Loading and preprocessing the EMNIST 'balanced' dataset.
- Building, compiling, and training a sequential neural network.
- Evaluating the model's performance on the test set.
- Visualizing the training history.


## 📂 Dataset
- **Dataset**: EMNIST (Extended MNIST) - Balanced Split  
- **Training Set**: 112,800 images  
- **Testing Set**: 18,800 images  
- **Classes**: 47 balanced classes  
- **Image Size**: 28x28 grayscale images  


## ⚙️ Preprocessing Steps
- **Normalization**: Pixel values scaled to range **[0, 1]**  
- **Flattening**: Each 28x28 image flattened into a **784-dimensional vector**  
- **One-Hot Encoding**: Integer labels converted into one-hot encoded format  



## 🧠 Model Architecture
The final model is a **Sequential Neural Network** with the following layers:

1. **Input Layer**: 784-dimensional vector (flattened image)  
2. **Dense Layer**: 2048 neurons, ReLU activation  
   - Batch Normalization  
   - Dropout (0.5)  
3. **Dense Layer**: 1024 neurons, ReLU activation  
   - Batch Normalization  
   - Dropout (0.5)  
4. **Dense Layer**: 512 neurons, ReLU activation  
   - Batch Normalization  
   - Dropout (0.5)  
5. **Output Layer**: 47 neurons, Softmax activation (multi-class classification)  



## 🏋️ Training
- **Optimizer**: Adam with ExponentialDecay learning rate schedule  
- **Loss Function**: `categorical_crossentropy`  
- **Metrics**: Accuracy  
- **Epochs**: 20 (with EarlyStopping)  
- **Batch Size**: 32  
- **Callbacks**: EarlyStopping (monitored `val_accuracy`) to prevent overfitting  



## 📊 Results
- **Final Test Accuracy**: **86.28%**  

Training and Validation Accuracy plots were generated during training.  
*(Notebook contains visualization plots.)*

---

## 🚀 How to Run
### Prerequisites
Install the required dependencies:
```bash
pip install tensorflow tensorflow_datasets matplotlib numpy
