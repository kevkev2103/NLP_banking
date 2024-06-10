from pydantic import Field
from langchain.chains.base import Chain
from datetime import datetime
import pymongo
from pymongo import MongoClient
from dotenv import load_dotenv
import os
from .models import SpeechToTextModel, TranslationModel, SentimentAnalysisModel, GenerateResponseModel
from new import speak_to_microphone

# Chargement des variables d'environnement
load_dotenv()
speech_key = os.getenv("api_key")
service_region = os.getenv("region")
mongodb_uri = os.getenv("mongodb_uri")

# Vérification que toutes les variables d'environnement sont bien présentes
if not all([speech_key, service_region, mongodb_uri]):
    raise ValueError("One or more environment variables are missing")

# Configuration de MongoDB
client = MongoClient(mongodb_uri)
db = client["AzureSpeechToText"]
collection = db["transcriptions"]

class GenerateResponseChain(Chain):
    stt_model: SpeechToTextModel = Field(...)
    translation_model: TranslationModel = Field(...)
    sentiment_model: SentimentAnalysisModel = Field(...)
    generate_response_model: GenerateResponseModel = Field(...)
    collection: pymongo.collection.Collection = Field(...)

    def __init__(self, stt_model, translation_model, sentiment_model, generate_response_model, collection, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stt_model = stt_model
        self.translation_model = translation_model
        self.sentiment_model = sentiment_model
        self.generate_response_model = generate_response_model
        self.collection = collection

    def _call(self, inputs):
        audio_file_path = inputs['audio_file_path']
        output_file = inputs.get('output_file', 'transcriptions/audio_wav_transcript_translate.txt')
        
        # Étape 1 : Transcription
        transcription = self.stt_model.transcribe(audio_file_path, output_file)
        # Étape 2 : Traduction
        translation = self.translation_model.translate(transcription)
        # Étape 3 : Analyse de Sentiment
        sentiment = self.sentiment_model.analyze_sentiment(translation)['label']
        # Étape 4 : Génération de Réponse
        response = self.generate_response_model.generate_response(sentiment, translation)
        # Étape 5 : Génération de date
        timestamp = datetime.now()
        
        # Sauvegarder la sortie de chaque étape dans MongoDB
        self.collection.insert_one({
            "transcription": transcription,
            "translation": translation,
            "timestamp": timestamp,
            "sentiment": sentiment,
            "response": response
        })
        
        return {
            'transcription': transcription,
            'translation': translation,
            'timestamp': timestamp,
            'sentiment': sentiment,
            'response': response
        }

    def process(self, inputs):
        return self._call(inputs)

    @property
    def input_keys(self):
        return ['audio_file_path']
    
    @property
    def output_keys(self):
        return ['transcription', 'translation', 'timestamp', 'sentiment', 'response']

# Génération des modèles
stt_model = speak_to_microphone()
translation_model = TranslationModel()
sentiment_model = SentimentAnalysisModel()
generate_response_model = GenerateResponseModel()

# Création de la chaîne
chain = GenerateResponseChain(
    stt_model=stt_model,
    translation_model=translation_model,
    sentiment_model=sentiment_model,
    generate_response_model=generate_response_model,
    collection=collection
)

# Fichier audio en entrée
input_audio = "audio_files/converted_2.wav"

# Exécution de la chaîne
results = chain.process({'audio_file_path': input_audio})
print(results)