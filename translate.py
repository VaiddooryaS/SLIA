from googletrans import Translator

def translate_text(text, target_lang):
    translator = Translator()
    translated = translator.translate(text, dest=target_lang)
    return translated.text

# print(translate_text("Hello", "ta"))  # Translate to Tamil
