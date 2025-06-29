# IMAGE-CLASSIFICATION-MODEL

*COMPANY* : CODTECH IT SOLUTIONS

*NAME* : GHADIYARAM JAYA SAI SREE RAMA KUMAR

*INTERN ID* : CT06DM1188

*DOMAIN* : MACHINE LEARNING

*DURATION* : 6 WEEKS

*MENTOR* : NEELA SANTOSH

This project is a practical implementation of an image classification system using Convolutional Neural Networks (CNNs) on the well-known CIFAR-10 dataset. The goal of the project is to accurately classify color images into one of ten distinct categories: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. Each image in the dataset is a 32x32 pixel color image, making it a manageable yet challenging dataset due to its relatively small size and high inter-class similarity. This project walks through the entire deep learning pipeline, from loading the dataset to building, training, evaluating, and visualizing the performance of the model.

To begin, essential Python libraries such as TensorFlow, NumPy, Matplotlib, and Seaborn are imported. TensorFlow is used for building and training the neural network, while NumPy handles numerical operations and Matplotlib/Seaborn help visualize the results. The CIFAR-10 dataset is loaded directly from TensorFlow’s datasets module, which conveniently provides pre-split training and testing data. This saves time and ensures standardization across projects.

Once loaded, the data is visually inspected. A small sample of the training images is plotted using matplotlib.pyplot along with their corresponding class names. This initial visualization is helpful to understand the kind of data being dealt with and confirms the integrity of the dataset. The dataset contains 60,000 images in total—50,000 for training and 10,000 for testing—with each image labeled with an integer corresponding to one of the ten classes.

Next, a Convolutional Neural Network (CNN) is defined using TensorFlow’s Keras API. CNNs are particularly effective for image recognition tasks because they are able to capture spatial hierarchies in images through the use of convolutional layers. The architecture of the model consists of three main convolutional blocks. Each block begins with a Conv2D layer using ReLU activation followed by a MaxPooling2D layer to downsample the feature maps and reduce computation. These are followed by a flattening step to convert the 2D feature maps into a 1D vector.

The flattened output is passed through a dense layer with 64 neurons using ReLU activation, and finally through an output layer with 10 neurons using the softmax activation function to output a probability distribution across the 10 classes. The model is compiled with the Adam optimizer, which adapts the learning rate during training, and uses sparse_categorical_crossentropy as the loss function since the labels are integer encoded. The model also tracks accuracy as a metric during training.

Training is done using the fit() function for 10 epochs, using the training set for learning and the testing set for validation. After training, the model is evaluated on the test data using model.evaluate() to determine its final accuracy and loss. The result is printed to get a quick understanding of how well the model generalizes to unseen data.

To better understand how the model performed during training, a visualization is created that plots the training and validation accuracy and loss over each epoch. This helps identify whether the model overfit the training data or if it improved steadily. By plotting both training and test metrics side-by-side, it's easier to spot patterns like overfitting, underfitting, or convergence.

In conclusion, this project showcases a foundational deep learning pipeline using CNNs for image classification. It demonstrates all major components of an image classification task—data loading and visualization, CNN model design, training, evaluation, and result analysis. Despite being relatively simple in structure, this project is powerful and serves as a strong introduction to image classification using deep learning. It can be further improved by adding data augmentation, dropout layers to reduce overfitting, or experimenting with more complex architectures like ResNet or VGGNet. This project is an excellent stepping stone for more advanced work in computer vision and deep learning.

![Image](https://github.com/user-attachments/assets/18f1b98a-c7b7-4035-9723-3c3667ea51c3)

![Image](https://github.com/user-attachments/assets/82d9082a-5cdc-4b77-9778-f823330248c7)

![Image](https://github.com/user-attachments/assets/a2b57e74-4f59-4768-8e5b-11392f43a35d)

![Image](https://github.com/user-attachments/assets/d56ae1d5-f5f9-41ec-8019-b3c246ef5f00)
