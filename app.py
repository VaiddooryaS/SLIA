import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import tensorflow as tf
import threading
import time
import os
import pygame
from gtts import gTTS
from translate import translate_text

# Initialize pygame mixer for speech output
pygame.mixer.init()

# Load Pretrained Model
MODEL_PATH = "models/sign_language_model1.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Initialize Mediapipe Hand Detector
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.8, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils


class SLIAApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SLIA - Sign Language Interpreter")
        self.root.geometry("1200x700")
        self.root.configure(bg="white")

        # Title
        self.title_label = tk.Label(self.root, text="SLIA - Sign Language Interpreter", font=("Arial", 18, "bold"), bg="white")
        self.title_label.pack(pady=10)

        # Camera Frame
        self.video_label = tk.Label(self.root, bg="black")
        self.video_label.place(x=20, y=60, width=600, height=500)

        # Predicted Letter UI
        self.letter_var = tk.StringVar()
        self.predicted_label = tk.Label(self.root, text="Predicted Letter:", font=("Arial", 14, "bold"), bg="white")
        self.predicted_label.place(x=640, y=60)

        self.letter_display = tk.Label(self.root, textvariable=self.letter_var, font=("Arial", 20, "bold"), fg="blue", bg="white")
        self.letter_display.place(x=820, y=60)

        # Sentence Display
        self.sentence_var = tk.StringVar()
        self.word_label = tk.Label(self.root, text="Sentence:", font=("Arial", 14, "bold"), bg="white")
        self.word_label.place(x=640, y=100)

        self.sentence_display = tk.Label(self.root, textvariable=self.sentence_var, font=("Arial", 14), fg="green", bg="white", wraplength=500, justify="left")
        self.sentence_display.place(x=640, y=130, width=500, height=80)

        # Buttons
        self.clear_button = tk.Button(self.root, text="CLEAR", bg="red", fg="white", font=("Arial", 12), command=self.clear_sentence)
        self.clear_button.place(x=640, y=230, width=80, height=40)

        self.space_button = tk.Button(self.root, text="SPACE", font=("Arial", 12), command=self.add_space)
        self.space_button.place(x=730, y=230, width=80, height=40)

        self.backspace_button = tk.Button(self.root, text="BACKSPACE", font=("Arial", 12), command=self.backspace)
        self.backspace_button.place(x=820, y=230, width=100, height=40)

        self.translate_button = tk.Button(self.root, text="TRANSLATE", font=("Arial", 12), command=self.translate_sentence)
        self.translate_button.place(x=940, y=230, width=100, height=40)

        self.speak_button = tk.Button(self.root, text="SPEAK", font=("Arial", 12), command=self.speak_sentence)
        self.speak_button.place(x=1060, y=230, width=80, height=40)

        # Language Dropdown
        self.language_label = tk.Label(self.root, text="Choose Language:", font=("Arial", 12), bg="white")
        self.language_label.place(x=640, y=300)

        self.language_var = tk.StringVar(value="hindi")
        self.language_dropdown = ttk.Combobox(self.root, textvariable=self.language_var, values=["hindi", "tamil", "malayalam", "kannada", "telugu"])
        self.language_dropdown.place(x=800, y=300, width=150, height=30)

        # Translated Text
        self.translated_text_var = tk.StringVar()
        self.translated_label = tk.Label(self.root, text="Translated Text:", font=("Arial", 14, "bold"), bg="white")
        self.translated_label.place(x=640, y=350)

        self.translated_display = tk.Label(self.root, textvariable=self.translated_text_var, font=("Arial", 16), fg="purple", bg="white", wraplength=500, justify="left")
        self.translated_display.place(x=640, y=380, width=500, height=80)

        # Video Processing Variables
        self.sentence = ""
        self.last_prediction_time = time.time()
        self.letter_display_time = time.time()
        self.prediction_delay = 2

        # Start Video Thread
        self.video_thread = threading.Thread(target=self.start_video)
        self.video_thread.daemon = True
        self.video_thread.start()

    def start_video(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            h, w, _ = frame.shape

            box_width = 200
            box_height = 200
            x_min = w - box_width - 30
            y_min = (h - box_height) // 2
            x_max = x_min + box_width
            y_max = y_min + box_height

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            hand_inside_box = False

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    hand_x = [lm.x * w for lm in hand_landmarks.landmark]
                    hand_y = [lm.y * h for lm in hand_landmarks.landmark]

                    if all(x_min < x < x_max for x in hand_x) and all(y_min < y < y_max for y in hand_y):
                        hand_inside_box = True

            if hand_inside_box and time.time() - self.last_prediction_time >= self.prediction_delay:
                self.last_prediction_time = time.time()

                hand_img = frame[y_min:y_max, x_min:x_max]
                hand_img = cv2.resize(hand_img, (64, 64))
                hand_img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)
                hand_img = np.expand_dims(hand_img, axis=[0, -1]) / 255.0

                prediction = model.predict(hand_img)
                predicted_letter = chr(np.argmax(prediction) + ord('A'))

                self.letter_display_time = time.time()
                self.letter_var.set(predicted_letter)
                self.sentence += predicted_letter
                self.sentence_var.set(self.sentence)

            if time.time() - self.letter_display_time > 5:
                self.letter_var.set("")

            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

    def clear_sentence(self):
        self.sentence = ""
        self.sentence_var.set("")
        self.translated_text_var.set("")

    def add_space(self):
        self.sentence += " "
        self.sentence_var.set(self.sentence)

    def backspace(self):
        self.sentence = self.sentence[:-1]
        self.sentence_var.set(self.sentence)

    def translate_sentence(self):
        text = self.sentence_var.get()
        target_lang = self.language_var.get()
        self.translated_text_var.set(translate_text(text, target_lang))

    def speak_sentence(self):
        text = self.translated_text_var.get().strip() or self.sentence_var.get().strip()
        if text:
            threading.Thread(target=lambda: self.play_speech(text)).start()

    def play_speech(self, text):
        try:
            pygame.mixer.music.stop()
            filename = "speech_output.mp3"
            tts = gTTS(text=text, lang="en")
            tts.save(filename)
            pygame.mixer.music.load(filename)
            pygame.mixer.music.play()
        except Exception as e:
            print(f"Error in speech synthesis: {e}")


# Run GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = SLIAApp(root)
    root.mainloop()