
# Handwritten Digit Recognition ✍️

## Project Statement
Recognize handwritten digits.

## Description

Handwritten digit recognition is the foundation of Optical Character Recognition (OCR) technology. Using TensorFlow and Python, you'll build a digit recognition model based on neural networks. This project is an excellent introduction to image classification.

![Handwritten Digit](image.png)

*Picture credits: [Handwritten Digit Classification with Arduino and MicroML](https://eloquentarduino.github.io/2020/02/handwritten-digit-classification-with-arduino-and-microml/)*

## Tools 🛠️

- TensorFlow
- Python

## Concepts 📚

- Image Classification
- Neural Networks

## Difficulty Level ⭐⭐

## Project Setup and Implementation

### Step 1: Setup the Environment

1. **Install Required Libraries:**
   ```bash
   pip install tensorflow numpy matplotlib
   ```

2. **Import Libraries:**
   ```python
   import tensorflow as tf
   from tensorflow.keras import datasets, layers, models
   import matplotlib.pyplot as plt
   import numpy as np
   ```

### Step 2: Load and Prepare the Dataset

1. **Load the MNIST Dataset:**
   ```python
   (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

   # Normalize the images to [0, 1] range
   train_images, test_images = train_images / 255.0, test_images / 255.0

   # Expand dimensions to fit Conv2D layer requirements
   train_images = np.expand_dims(train_images, axis=-1)
   test_images = np.expand_dims(test_images, axis=-1)
   ```

2. **Explore the Data:**
   ```python
   # Display the first 25 images from the training set and their labels
   plt.figure(figsize=(10,10))
   for i in range(25):
       plt.subplot(5, 5, i+1)
       plt.xticks([])
       plt.yticks([])
       plt.grid(False)
       plt.imshow(train_images[i], cmap=plt.cm.binary)
       plt.xlabel(train_labels[i])
   plt.show()
   ```

### Step 3: Build the Improved Model with Dropout

1. **Define the Model Architecture:**
   ```python
   model = models.Sequential([
       layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
       layers.MaxPooling2D((2, 2)),
       layers.Dropout(0.25),
       layers.Conv2D(64, (3, 3), activation='relu'),
       layers.MaxPooling2D((2, 2)),
       layers.Dropout(0.25),
       layers.Conv2D(64, (3, 3), activation='relu'),
       layers.Flatten(),
       layers.Dense(64, activation='relu'),
       layers.Dropout(0.5),
       layers.Dense(10, activation='softmax')
   ])

   model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
   ```

2. **Train the Model:**
   ```python
   history = model.fit(train_images, train_labels, epochs=10, 
                       validation_data=(test_images, test_labels))
   ```

### Step 4: Evaluate the Improved Model

1. **Evaluate the Model:**
   ```python
   test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
   print(f'\nTest accuracy: {test_acc}')
   ```

2. **Visualize Training History:**
   ```python
   # Plot training & validation accuracy values
   plt.figure(figsize=(12, 4))
   plt.subplot(1, 2, 1)
   plt.plot(history.history['accuracy'], label='Train Accuracy')
   plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
   plt.title('Model Accuracy')
   plt.xlabel('Epoch')
   plt.ylabel('Accuracy')
   plt.legend(loc='upper left')

   # Plot training & validation loss values
   plt.subplot(1, 2, 2)
   plt.plot(history.history['loss'], label='Train Loss')
   plt.plot(history.history['val_loss'], label='Validation Loss')
   plt.title('Model Loss')
   plt.xlabel('Epoch')
   plt.ylabel('Loss')
   plt.legend(loc='upper left')

   plt.show()
   ```

### Step 5: Make Predictions

1. **Make Predictions:**
   ```python
   predictions = model.predict(test_images)

   # Display the first test image, its predicted label, and the true label
   def plot_image(i, predictions_array, true_label, img):
       true_label, img = true_label[i], img[i]
       plt.grid(False)
       plt.xticks([])
       plt.yticks([])
       plt.imshow(img, cmap=plt.cm.binary)

       predicted_label = np.argmax(predictions_array)
       if predicted_label == true_label:
           color = 'blue'
       else:
           color = 'red'

       plt.xlabel(f"{predicted_label} ({true_label})", color=color)

   def plot_value_array(i, predictions_array, true_label):
       true_label = true_label[i]
       plt.grid(False)
       plt.xticks(range(10))
       plt.yticks([])
       thisplot = plt.bar(range(10), predictions_array, color="#777777")
       plt.ylim([0, 1])
       predicted_label = np.argmax(predictions_array)

       thisplot[predicted_label].set_color('red')
       thisplot[true_label].set_color('blue')

   i = 0
   plt.figure(figsize=(6,3))
   plt.subplot(1,2,1)
   plot_image(i, predictions[i], test_labels, test_images)
   plt.subplot(1,2,2)
   plot_value_array(i, predictions[i], test_labels)
   plt.show()
   ```

## Results

The model achieved a test accuracy of approximately `XX%`. The figure below shows an example of the model's predictions:

![Prediction Example](image.png)

## How to Use this Repo

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Install the required libraries:**
   ```bash
   pip install tensorflow numpy matplotlib
   ```

3. **Run the Jupyter notebook or Python script:**
   ```bash
   jupyter notebook Handwritten_Digit_Recognition.ipynb
   # OR
   python Handwritten_Digit_Recognition.py
   ```

## Contributing

Contributions are welcome! Please fork this repository and submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- MNIST dataset: [Yann LeCun and Corinna Cortes](http://yann.lecun.com/exdb/mnist/)
- TensorFlow: [TensorFlow](https://www.tensorflow.org/)
- Matplotlib: [Matplotlib](https://matplotlib.org/)
