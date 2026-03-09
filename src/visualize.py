# visualize.py
import numpy as np
import matplotlib.pyplot as plt
import os

# Ensure output folder exists
os.makedirs("../outputs/charts", exist_ok=True)

# Load analysis results
avg_intensity = np.load("../outputs/charts/avg_intensity.npy", allow_pickle=True)
classes = np.load("../outputs/charts/classes.npy", allow_pickle=True)

# Plot bar chart
plt.figure(figsize=(10,6))
plt.bar(classes, avg_intensity, color='skyblue')
plt.xticks(rotation=45)
plt.ylabel("Average Pixel Intensity")
plt.title("Fashion-MNIST: Average Intensity per Class")
plt.tight_layout()

# Save chart
chart_path = "../outputs/charts/fashion_mnist_intensity.png"
plt.savefig(chart_path)
plt.show()

print(f"Chart saved at {chart_path}")