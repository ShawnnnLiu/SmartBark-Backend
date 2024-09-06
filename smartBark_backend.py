# Saving the code from the images into a Python file

import numpy
import time
import pyaudio
#import analysyeffi
import wave
from multiprocessing import Process
import tensorflow as tf
import librosa
import numpy as np
import requests
import json
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

serverToken = 'AAAAVPxLM2A:APA91bHohk7YJYs_tXXiwLAyLFuuiqZkHjVmTks7BlYcLTuPVIOE77BP1syddnyIAqPBZpRKLyn8qIZhVDzAdvLf5nm5ge1zsuUYG5SHq0s-oKc64BTPDGUzSw2q6n8ITJ'
deviceToken = 'dW8nL_9T1Ca6wGeWUwG61rAPA91bFjmLO9y24u4X0xteXD6x43LD3-bCdKf9jf1B7ausqeb80vH-ZE703ovEYh0xJl0w-7rr37aGeUY10bnR9Y3SwIhk_khp81qmJr0GSVyYAtcRA_Fg'
headers = {'Content-Type': 'application/json', 'Authorization': 'key=' + serverToken,}

streamChunk = 1024  # chunk used for the audio input stream
sampleRate = 44100  # the sample rate of the user's mic
input_device_index = 0  # device index for the user's mic
numChannels = 1  # number of channels for the user's mic
audio_format = pyaudio.paInt16  # the audio format
ambient_db = -5  # the ambience noise level in db
email_timer = 15
seconds = 4

emailSentAt = None

p = pyaudio.PyAudio()

stream = pyaudio.PyAudio().open(
    format = audio_format,
    channels = numChannels,
    rate = sampleRate,
    input_device_index = input_device_index,
    input = True
)

print("Starting BarkTracker")

def saveToWav(frames):
    filename = "bark" + str(int(time.time())) + ".wav"
    wf = wave.open(filename, 'wb')
    wf.setnchannels(numChannels)
    wf.setsampwidth(pyaudio.get_sample_size(audio_format))
    wf.setframerate(sampleRate)
    wf.writeframes(b''.join(frames))
    wf.close()
    return filename

sound_dict = { 0: 'air_conditioner', 1: 'car_horn', 2: 'dog_bark', 3: 'children_playing', 4: 'drilling', 5: 'engine_idling',
               6: 'gun_shot', 7: 'jackhammer', 8: 'siren', 9: 'street_music' }

model = tf.keras.models.load_model("my_model.h5")

def get_prediction(model, wav_file):
    dat1, sample_rate = librosa.load(wav_file)
    mels = np.mean(librosa.feature.melspectrogram(y=dat1, sr=sample_rate).T, axis=0)
    arr = mels.reshape(1,16,8,1)
    pred = model.predict(arr)
    pred_index = np.argmax(pred, axis = 1)[0]
    return pred_index, pred[pred_index]

def get_device_tokens():
    cred = credentials.Certificate('service_account_key.json')
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    users_ref = db.collection(u'tokens')
    docs = users_ref.stream()
    device_tokens = []
    for doc in docs:
        device_tokens.append(doc.to_dict()['token'])
    return device_tokens

while True:
    time.sleep(2)
    #read new mic data
    frames = []
    for i in range(0, int(sampleRate / streamChunk * seconds)):
        data = stream.read(streamChunk, exception_on_overflow = False)
        frames.append(data)
    fileName = saveToWav(frames)
    print("Done")
    
    #convert to NumPy array
    rawsamps = stream.read(streamChunk, exception_on_overflow = False)
    samps = numpy.fromstring(rawsamps, dtype = numpy.int16)
    pred, confidence = get_prediction(model,fileName)
    record = {}
    print(sound_dict[pred])
    if pred == 2:
        record[str(int(time.time()))] = str(confidence)
        response = requests.put('https://dog-bark-detector.firebaseio.com/device/mobile/timestamp.json', data=json.dumps(record))
        deviceTokens = get_device_tokens()
        for token in deviceTokens:
            body = {'notification': {'title': 'Notification from Dog Bark Detector','body': 'Your Dog Barked!'},'to': token, 'priority': 'high'}
            response = requests.post("https://fcm.googleapis.com/fcm/send",headers = headers, data=json.dumps(body))
            print(response)

