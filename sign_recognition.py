import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model("models/sign_language_model.h5")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils  # For drawing hand key points
hands = mp_hands.Hands(min_detection_confidence=0.8)
cap = cv2.VideoCapture(0)

# Define fixed bounding box on the right side
BOX_X_MIN = 400
BOX_Y_MIN = 100
BOX_X_MAX = 600
BOX_Y_MAX = 300

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame for a natural mirroring effect
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # Draw fixed bounding box
    cv2.rectangle(frame, (BOX_X_MIN, BOX_Y_MIN), (BOX_X_MAX, BOX_Y_MAX), (0, 255, 0), 2)

    # If hand landmarks are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw key points on hand
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract hand region inside the fixed bounding box
            hand_img = frame[BOX_Y_MIN:BOX_Y_MAX, BOX_X_MIN:BOX_X_MAX]
            hand_img = cv2.resize(hand_img, (64, 64))
            hand_img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)
            hand_img = np.expand_dims(hand_img, axis=[0, -1]) / 255.0  # Normalize

            # Make prediction
            prediction = model.predict(hand_img)
            predicted_label = chr(np.argmax(prediction) + ord('A'))

            # Display predicted text
            cv2.putText(frame, f"Prediction: {predicted_label}", (BOX_X_MIN, BOX_Y_MIN - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show frame
    cv2.imshow("Sign Language Recognition", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
