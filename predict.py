import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load trained model
model = tf.keras.models.load_model("digit_cnn.h5")

# Load MNIST test dataset
(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize and reshape
x_test = x_test / 255.0
x_test = x_test.reshape((-1, 28, 28, 1))

# Pick a random sample
index = np.random.randint(0, len(x_test))
sample = x_test[index]
true_label = y_test[index]

# Predict
prediction = model.predict(sample.reshape(1, 28, 28, 1))
predicted_label = np.argmax(prediction)

print(f"✅ Predicted: {predicted_label}, 🎯 Actual: {true_label}")

# Show the image
plt.imshow(sample.reshape(28, 28), cmap="gray")
plt.title(f"Predicted: {predicted_label}, Actual: {true_label}")
plt.show()
