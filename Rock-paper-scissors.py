import cv2
import numpy as np
import tensorflow as tf

# Load your trained model
model = tf.keras.models.load_model('model10.h5')

# Class labels
labels = ["rock", "paper", "scissors"]

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame
    img = cv2.resize(frame, (100, 100))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    prediction = model.predict(img)
    choice = labels[np.argmax(prediction)]

    # Show prediction on screen
    cv2.putText(frame, f"Prediction: {choice}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Rock Paper Scissors", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()