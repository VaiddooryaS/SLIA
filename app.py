import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import tensorflow as tf
import os
import threading
from translate import translate_text
from tts import text_to_speech

# Load Pretrained Model
MODEL_PATH = "models/sign_language_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Initialize Mediapipe Hand Detector
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.8, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils


# GUI Application
class SLIAApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sign Language Interpreter")
        self.root.geometry("900x600")
        self.root.configure(bg="white")

        # CAMERA FRAME
        self.video_label = tk.Label(self.root, bg="black")
        self.video_label.place(x=20, y=20, width=400, height=300)

        # PREDICTED LETTER
        self.letter_var = tk.StringVar()
        self.predicted_label = tk.Label(self.root, text="Predicted Letter:", font=("Arial", 14, "bold"), bg="white")
        self.predicted_label.place(x=450, y=20)

        self.letter_display = tk.Label(self.root, textvariable=self.letter_var, font=("Arial", 20, "bold"), fg="blue",
                                       bg="white")
        self.letter_display.place(x=620, y=20)

        # SUGGESTED WORDS
        self.suggestions_label = tk.Label(self.root, text="Suggestions:", font=("Arial", 12, "bold"), bg="white")
        self.suggestions_label.place(x=450, y=60)

        self.suggestion_buttons = []
        for i in range(3):
            btn = tk.Button(self.root, text=f"Option {i + 1}", font=("Arial", 12), bg="lightgray")
            btn.place(x=450 + (i * 100), y=90, width=80, height=30)
            self.suggestion_buttons.append(btn)

        # SENTENCE DISPLAY
        self.sentence_var = tk.StringVar()
        self.word_label = tk.Label(self.root, text="Sentence:", font=("Arial", 14, "bold"), bg="white")
        self.word_label.place(x=450, y=140)

        self.sentence_display = tk.Label(self.root, textvariable=self.sentence_var, font=("Arial", 16), fg="green",
                                         bg="white")
        self.sentence_display.place(x=450, y=170)

        # BUTTONS
        self.clear_button = tk.Button(self.root, text="CLEAR", bg="red", fg="white", font=("Arial", 12),
                                      command=self.clear_sentence)
        self.clear_button.place(x=450, y=210, width=80, height=40)

        self.translate_button = tk.Button(self.root, text="TRANSLATE", font=("Arial", 12),
                                          command=self.translate_sentence)
        self.translate_button.place(x=550, y=210, width=100, height=40)

        self.speak_button = tk.Button(self.root, text="SPEAK", font=("Arial", 12), command=self.speak_sentence)
        self.speak_button.place(x=670, y=210, width=80, height=40)

        # LANGUAGE DROPDOWN
        self.language_label = tk.Label(self.root, text="Choose Language:", font=("Arial", 12), bg="white")
        self.language_label.place(x=450, y=270)

        self.language_var = tk.StringVar(value="hindi")
        self.language_dropdown = ttk.Combobox(self.root, textvariable=self.language_var,
                                              values=["hindi", "tamil", "malayalam", "kannada", "telugu"])
        self.language_dropdown.place(x=600, y=270, width=150, height=30)

        # TRANSLATED TEXT
        self.translated_text_var = tk.StringVar()
        self.translated_label = tk.Label(self.root, text="Translated Text:", font=("Arial", 14, "bold"), bg="white")
        self.translated_label.place(x=450, y=320)

        self.translated_display = tk.Label(self.root, textvariable=self.translated_text_var, font=("Arial", 16),
                                           fg="purple", bg="white")
        self.translated_display.place(x=450, y=350)

        # Start Video Thread
        self.sentence = ""
        self.video_thread = threading.Thread(target=self.start_video)
        self.video_thread.daemon = True
        self.video_thread.start()

    def start_video(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            h, w, c = frame.shape
            x_min, y_min, x_max, y_max = int(w * 0.3), int(h * 0.3), int(w * 0.7), int(h * 0.7)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    hand_img = frame[y_min:y_max, x_min:x_max]
                    hand_img = cv2.resize(hand_img, (64, 64))
                    hand_img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)
                    hand_img = np.expand_dims(hand_img, axis=[0, -1]) / 255.0

                    prediction = model.predict(hand_img)
                    predicted_label = chr(np.argmax(prediction) + ord('A'))
                    self.letter_var.set(predicted_label)
                    self.sentence += predicted_label
                    self.sentence_var.set(self.sentence)

            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

    def clear_sentence(self):
        self.sentence = ""
        self.sentence_var.set("")
        self.translated_text_var.set("")

    def translate_sentence(self):
        text = self.sentence_var.get()
        target_lang = self.language_var.get()
        translated_text = translate_text(text, target_lang)
        self.translated_text_var.set(translated_text)

    def speak_sentence(self):
        text = self.translated_text_var.get() or self.sentence_var.get()
        text_to_speech(text)


# Run GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = SLIAApp(root)
    root.mainloop()
