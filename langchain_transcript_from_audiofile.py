import os
from dotenv import load_dotenv
import time
import threading
import pymongo
from pymongo import MongoClient
import azure.cognitiveservices.speech as speechsdk
from pydantic import BaseModel, Field
from datetime import datetime

class SpeechToText:
    def __init__(self):
        load_dotenv()
        self.speech_key = os.getenv("api_key")
        self.service_region = os.getenv("region")
        self.mongodb_uri = os.getenv("mongodb_uri")

        if not all([self.speech_key, self.service_region, self.mongodb_uri]):
            raise ValueError("One or more environment variables are missing")

        # MongoDB setup
        self.cluster = MongoClient(self.mongodb_uri)
        self.db = self.cluster["AzureSpeechToText"]
        self.collection = self.db["transcriptions"]

    def transcribe(self, audio_file_path, output_file):
        if not os.path.isfile(audio_file_path):
            raise FileNotFoundError(f"Audio file not found: {audio_file_path}")

        speech_config = speechsdk.SpeechConfig(subscription=self.speech_key, region=self.service_region)
        speech_config.speech_recognition_language = "fr-FR"
        
        audio_config = speechsdk.audio.AudioConfig(filename=audio_file_path)
        recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

        print("Reconnaissance vocale en cours à partir du fichier audio...")

        all_results = []

        done = threading.Event()

        def recognized(evt):
            all_results.append(evt.result.text)
            print(f"Texte reconnu: {evt.result.text}")

        def stop_cb(evt):
            print('CLOSING on {}'.format(evt))
            done.set()

        recognizer.recognized.connect(recognized)
        recognizer.session_stopped.connect(stop_cb)
        recognizer.canceled.connect(stop_cb)

        recognizer.start_continuous_recognition()
        done.wait()

        recognizer.stop_continuous_recognition()
        
        with open(output_file, "w", encoding='utf-8') as file:
            file.write(" ".join(all_results))
        timestamp = datetime.now()
        self.collection.insert_one({"transcription": " ".join(all_results), "timestamp": timestamp})

from langchain.chains.base import Chain
from langchain.chains import LLMChain

class TranscriptionChain(Chain):
    stt_model: SpeechToText = Field(...)

    def __init__(self, stt_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stt_model = stt_model

    def _call(self, inputs):
        audio_file_path = inputs["audio_file_path"]
        output_file = inputs.get('output_file', 'transcriptions/audio_wav_transcript.txt')
        self.stt_model.transcribe(audio_file_path, output_file)
        with open(output_file, 'r', encoding='utf-8') as file:
            transcription = file.read()
        return {'transcription': transcription}
    
    @property
    def input_keys(self):
        return ['audio_file_path']
    
    @property
    def output_keys(self):
        return ['transcription']

# Utilisation du model avec LangChain
transcription_chain = TranscriptionChain(stt_model=SpeechToText())
audio_file_path = 'audio_files/output.wav'
result = transcription_chain({'audio_file_path': audio_file_path})

print(result['transcription'])