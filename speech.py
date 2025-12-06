from text_to_speech import save
from playsound3 import playsound
from threading import Thread
import tempfile
import os

def _tts(text, language="en"):
    with tempfile.TemporaryDirectory() as temp_dir:
        filepath = os.path.join(temp_dir, "audio.mp3")
        save(text, language, file=filepath)
        playsound(filepath)

def tts(text, language="en"):
    thread = Thread(target=_tts, args=(text, language))
    thread.start()

if __name__ == "__main__":
    text = "Hello World!"
    tts(text)