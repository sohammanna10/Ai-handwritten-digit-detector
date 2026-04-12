import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# Load trained model
model = tf.keras.models.load_model("digit_cnn.h5")

# Create a blank image to draw
canvas = np.zeros((280, 280), dtype=np.uint8)

def predict_digit(img):
    img = cv2.resize(img, (28,28))
    img = img / 255.0
    img = img.reshape(1,28,28,1)
    prediction = model.predict(img)
    return np.argmax(prediction)

def draw_and_predict():
    drawing = False
    last_point = None

    def draw(event, x, y, flags, param):
        nonlocal drawing, last_point
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            last_point = (x,y)
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                cv2.line(canvas, last_point, (x,y), 255, 20)
                last_point = (x,y)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False

    cv2.namedWindow("Draw Digit")
    cv2.setMouseCallback("Draw Digit", draw)

    while True:
        cv2.imshow("Draw Digit", canvas)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("c"):   # Clear canvas
            canvas[:] = 0
        elif key == ord("q"): # Quit
            break

    cv2.destroyAllWindows()
    digit = predict_digit(canvas)
    plt.imshow(canvas, cmap="gray")
    plt.title(f"Predicted: {digit}")
    plt.show()

draw_and_predict()
