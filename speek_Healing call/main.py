import requests
import speech_recognition as sr

# Create a Recognizer instance
recognizer = sr.Recognizer()

# Use the default microphone as the audio source
with sr.Microphone() as source:
    print("Adjusting for ambient noise... Please wait.")
    recognizer.adjust_for_ambient_noise(source)

    print("Say something!")
    audio = recognizer.listen(source)

    print("Recognizing...")

    try:
        # Use Google Web Speech API
        word = recognizer.recognize_google(audio)
        print("You said:", word)

    except sr.UnknownValueError:
        print("Sorry, could not understand your speech.")
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))

response = requests.post("http://127.0.0.1:5000/predict",
                         json={"text": [word]})
print(response.json())

