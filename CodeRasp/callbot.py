import speech_recognition as sr
import pyaudio
from communicate_firebase import *
import sys
import os
import gtts
import time
from multiprocessing import Process
from threading import Thread
import sys
import wikipedia
from queue import Queue
wikipedia.set_lang('vi')
que1=Queue()
que2=Queue()
que3=Queue()
class Sound(object):
    def __init__(self) :
        self.v=1
        
    def sound(self,label):
        try:
            tts=gtts.gTTS(label,lang="vi")
            tts.save("hello.mp3")
            os.system("mpg123 hello.mp3")
            # os.remove("hello.mp3")
        except Exception as e:
            print("file is deleted")
    def speak(self,label):
        process=Thread(target=self.sound, args=(label,))
        process.start()
        if 0xff==ord("q"):
            sys.exit()
sound=Sound()
def get_audio():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Tôi: ", end='')
        audio = r.listen(source, phrase_time_limit=5)
        try:
            text = r.recognize_google(audio, language="vi-VN")
            print(text)
            return text
        except:
            print("...")
            return 0
def stop():
    sound.speak("Hẹn gặp lại bạn sau!")
def get_text():
    for i in range(3):
        text = get_audio()
        if text:
            return text.lower()
        elif i < 2:
            print("Bot không nghe rõ. Bạn nói lại được không!")
    time.sleep(2)
    stop()
    return 0
def tell_me_about():
    try:
        sound.speak("Bạn muốn nghe về gì ạ")
        text = get_text()
        contents = wikipedia.summary(text).split('\n')
        sound.speak(contents[0])
        time.sleep(10)
        for content in contents[1:]:
            sound.speak("Bạn muốn nghe thêm không")
            ans = get_text()
            if "có" not in ans:
                break    
            sound.speak(content)
            time.sleep(10)

        sound.speak('Cảm ơn bạn đã lắng nghe!!!')
    except:
        sound.speak("Bot không định nghĩa được thuật ngữ của bạn. Xin mời bạn nói lại")
def control():
    Thread_read_data(ref_obj,que1)
    Thread_read_data(ref_human,que2)
    if que1.qsize() >0 or que2.qsize() >0: 
        data_obj=que1.get()
        data_human=que2.get()
        sound.speak(data_human)
    text=get_text()
    if "xin chào" in text:
        data='A'
        Thread_update(ref_request,{data})
        sound.speak("Đã bật nhận diện vật thể")
    elif "hello" in text :
        data="B"
        Thread_update(ref_request,{data})
        sound.speak("Đã bật nhận diện chữ cái")
    elif "tắt" in text :
        data="S"
        Thread_update(ref_request,{data})
        sound.speak("Đã tắt nhận diện ")
    elif "hỏi" in text :
        tell_me_about()

while True:
    control()