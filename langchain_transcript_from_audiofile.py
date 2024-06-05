import os
from dotenv import load_dotenv
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
        # timestamp = datetime.now()
        # self.collection.insert_one({"transcription": " ".join(all_results), "timestamp": timestamp})
        return " ".join(all_results)

from langchain.chains.base import Chain
from langchain.chains import LLMChain
from transformers import pipeline

class TranslationModel:
    def __init__(self):
        self.translation_pipeline = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")

    def translate(self, text):
        result = self.translation_pipeline(text)
        return result[0]['translation_text'] # Comme le pipeline retourne une liste de dictionnaires, result[0] accède au premier dictionnaire de cette liste.


class Speech2TextTranslationChain(Chain):
    stt_model: SpeechToText = Field(...)
    translation_model = TranslationModel = Field(...)
    collection: pymongo.collection.Collection = Field(init=False)


    def __init__(self, stt_model, translation_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stt_model = stt_model
        self.translation_model = translation_model
        self.collection = self.stt_model.collection

    def _call(self, inputs):
        audio_file_path = inputs['audio_file_path']
        output_file = inputs.get('output_file', 'transcriptions/audio_wav_transcript_translate.txt')
        transcription = self.stt_model.transcribe(audio_file_path, output_file)
        translation = self.translation_model.translate(transcription)
        # timestamp = datetime.now()
        # self.collection.insert_one({"transcription": transcription, "translation":translation, "timestamp": timestamp})
        return {'transcription':transcription, 'translation':translation}
    
    @property # Ce décorateur est utilisé pour définir une méthode qui peut être accédée comme un attribut. En d'autres termes, il permet de définir des getters pour les attributs calculés.
    def input_keys(self): # Ces méthodes définissent les clés d'entrée et de sortie de votre chaîne de traitement. Elles spécifient les noms des paramètres que la chaîne attend en entrée et les noms des résultats que la chaîne produira en sortie.
        return ['audio_file_path'] # retourne une liste avec une seule chaîne de caractères 
        # Cela signifie que la chaîne attendra un dictionnaire d'entrée avec une clé nommée audio_file_path.
    
    @property
    def output_keys(self):
        return ['transcription', 'translation'] 
    
    '''
    Ces propriétés sont importantes pour définir de manière explicite l'interface de votre chaîne,
    c'est-à-dire ce qu'elle attend en entrée et ce qu'elle produit en sortie.
    Cela permet à LangChain de gérer les chaînes de traitement de manière cohérente et modulable.
    '''
    
# Utilisation du modèle avec LangChain
stt_model = SpeechToText()
translation_model = TranslationModel()
chain = Speech2TextTranslationChain(stt_model=stt_model, translation_model=translation_model)

audio_file_path = 'audio_files/output.wav'
result = chain({'audio_file_path': audio_file_path})

print("Transcription:", result['transcription'])
print("Translation:", result['translation'])




class SentimentAnalysisModel:
    def __init__(self):
        self.sentiment_pipeline = pipeline("text-classification", model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")

    def analyze_sentiment(self, text):
        result = self.sentiment_pipeline(text)
        return result[0]

class SentimentAnalysisChain(Chain):
    sentiment_model: SentimentAnalysisModel = Field(...)
    # translation_model = TranslationModel = Field(...)
    collection: pymongo.collection.Collection = Field(init=False)

    def __init__(self, sentiment_model, collection, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.translation_model = translation_model
        self.sentiment_model = sentiment_model
        self.collection = collection

    def _call(self, inputs):
        transcription = inputs["transcription"]
        translation = inputs["translation"]
        sentiment = self.sentiment_model.analyze_sentiment(translation)
        timestamp = datetime.now()
        self.collection.insert_one({"transcription": transcription, "translation":translation, "timestamp": timestamp, "sentiment":sentiment})
        return {'transcription': transcription, 'translation':translation, 'sentiment':sentiment}
    
    @property
    def input_keys(self):
        return ['transcription', 'translation']
    
    @property
    def output_keys(self):
        return ['transcription', 'translation', 'sentiment']
    


sentiment_model = SentimentAnalysisModel()
# Second chain: translation -> sentiment analysis
collection = stt_model.collection
sentiment_analysis_chain = SentimentAnalysisChain(sentiment_model=sentiment_model, collection=collection)
# result_sentiment = sentiment_analysis_chain({'translation': result_translation['translation']})
# Second chain: translation -> sentiment analysis
result_sentiment = sentiment_analysis_chain({
    'transcription': result['transcription'],
    'translation': result['translation']
})
print("Sentiment:", result_sentiment['sentiment'])