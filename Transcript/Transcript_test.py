import pyttsx3
from Bart_finetune.inference_test import infer

text = infer()
engine = pyttsx3.init()
engine.say(text)
engine.runAndWait()