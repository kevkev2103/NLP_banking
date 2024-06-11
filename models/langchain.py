from pydantic import Field
from langchain.chains.base import Chain
from datetime import datetime
import pymongo
from pymongo import MongoClient
from dotenv import load_dotenv
import os
import openai
from .models import Speech2TextModel, SentimentAnalysisModel, GenerateResponseModel

# Chargement des variables d'environnement
load_dotenv()
speech_key = os.getenv("speech_key")
region = os.getenv("region")
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.getenv("OPENAI_API_BASE")
openai.api_type = 'azure'
openai.api_version = os.getenv("OPENAI_VERSION")

# Configuration de MongoDB
mongodb_uri = os.getenv("mongodb_uri")
client = MongoClient(mongodb_uri)
db = client["AzureSpeechToText"]
collection = db["transcriptions"]

# Vérification que toutes les variables d'environnement sont bien présentes
if not all([speech_key, region, mongodb_uri]):
    raise ValueError("One or more environment variables are missing")

class DeepChain(Chain):
    stt_model: Speech2TextModel = Field(...)
    sentiment_model: SentimentAnalysisModel = Field(...)
    generate_response_model: GenerateResponseModel = Field(...)
    collection: pymongo.collection.Collection = Field(...)
    speech_key = Field(...)
    region = Field(...)
    mongodb_uri = Field(...)

    def __init__(self, stt_model, sentiment_model, generate_response_model, collection, speech_key, region, mongodb_uri, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stt_model = stt_model
        self.sentiment_model = sentiment_model
        self.generate_response_model = generate_response_model
        self.collection = collection
        load_dotenv()
        self.speech_key = os.getenv("speech_key")
        self.region = os.getenv("region")
        self.mongodb_uri = os.getenv("mongodb_uri")
        if not all([self.speech_key, self.region, self.mongodb_uri]):
            raise ValueError("One or more environment variables are missing")

    def _call(self, inputs):
        audio_file_path = inputs['audio_file_path']
        output_file = inputs.get('output_file', 'transcriptions/audio_wav_transcript_translate.txt')
        
        # Étape 1 : Transcription
        transcription = self.stt_model.speak_to_microphone(stop_phrase="stop session", output_file=output_file)
        # Étape 2 : Analyse de Sentiment
        sentiment = self.sentiment_model.analyze_sentiment([transcription])
        # Étape 3 : Génération de Réponse
        response = self.generate_response_model.generate_response(sentiment, transcription)
        # Étape 4 : Génération de date
        timestamp = datetime.now()
        
        # Sauvegarder la sortie de chaque étape dans MongoDB
        self.collection.insert_one({
            "transcription": transcription,
            "timestamp": timestamp,
            "sentiment": sentiment,
            "response": response
        })
        
        return {
            'transcription': transcription,
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
        return ['transcription', 'timestamp', 'sentiment', 'response']

# Génération des modèles
stt_model = Speech2TextModel()
sentiment_model = SentimentAnalysisModel()
generate_response_model = GenerateResponseModel()

# Création de la chaîne
chain = DeepChain(
    stt_model=stt_model,
    sentiment_model=sentiment_model,
    generate_response_model=generate_response_model,
    collection=collection,
    speech_key=speech_key,
    region = region,
    mongodb_uri = mongodb_uri
)

# Fichier audio en entrée
input_audio = "audio_files/converted_2.wav"

# Exécution de la chaîne
results = chain.process({'audio_file_path': input_audio})
print(results)