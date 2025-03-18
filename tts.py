from gtts import gTTS
import os

def text_to_speech(text):
    tts = gTTS(text=text, lang="en")
    tts.save("output.mp3")
    os.system("start output.mp3")  # Plays the speech output

text_to_speech("Hello, how are you?")
