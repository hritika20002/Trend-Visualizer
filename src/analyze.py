# analyze.py
import numpy as np
from tensorflow.keras.datasets import fashion_mnist

# Load Fashion-MNIST dataset
(x_train, y_train), (_, _) = fashion_mnist.load_data()

# Define class names
classes = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Calculate average intensity per class
avg_intensity = []
for i in range(10):
    images = x_train[y_train == i]
    avg_intensity.append(np.mean(images))

# Save results for visualization
np.save("../outputs/charts/avg_intensity.npy", avg_intensity)
np.save("../outputs/charts/classes.npy", classes)

print("Analysis complete. Data saved in outputs/charts/")