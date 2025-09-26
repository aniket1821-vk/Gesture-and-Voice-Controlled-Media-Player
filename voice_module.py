import speech_recognition as sr

class VoiceRecognition:
    def __init__(self):
        self.recognizer = sr.Recognizer()

    def get_command(self):
        with sr.Microphone() as source:
            print("Listening for 'hello' command...")
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            audio = self.recognizer.listen(source)
        
        try:
            print("Recognizing...")
            text = self.recognizer.recognize_google(audio)
            print("You said:", text)
            if "hello" in text.lower():
                return text
            else:
                print("Keyword 'hello' not detected.")
                return ""
        except sr.UnknownValueError:
            print("Could not understand audio.")
            return ""
        except sr.RequestError:
            print("Network error.")
            return ""
