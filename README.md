✍️ AI Handwritten Digit Detector

📌 Description

This project uses Machine Learning to recognize handwritten digits (0–9) in real time.
A trained neural network model predicts the digit based on input images or drawings.

🚀 Features

* Recognizes handwritten digits (0–9)
* Real-time prediction
* Custom drawing input support
* Trained using deep learning (TensorFlow/Keras)

🧠 How It Works

1. The model is trained on digit datasets
2. Input image is processed using OpenCV
3. Neural network predicts the digit
4. Output is displayed instantly

🛠️ Technologies Used

* Python
* OpenCV
* NumPy
* TensorFlow / Keras

▶️ How to Run

```bash
pip install -r requirements.txt
python predict.py
```

📂 Project Structure

```
.
├── train.py          # Train model
├── predict.py        # Predict digit
├── predict_draw.py   # Draw & predict
├── digit_cnn.h5      # Trained model
├── requirements.txt
└── README.md
```

🔮 Future Improvements

* Add GUI interface
* Improve accuracy
* Deploy as web app


👨‍💻 Author

Soham Manna
