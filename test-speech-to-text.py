import os
from dotenv import load_dotenv
import azure.cognitiveservices.speech as speechsdk
from pymongo import MongoClient
from transformers import pipeline


# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

subscription_key = os.getenv("AZURE_SUBSCRIPTION_KEY")
region = os.getenv("AZURE_REGION")
mongodb_uri = os.getenv("MONGODB_URI")
mongodb_db_name = os.getenv("MONGODB_DB_NAME")
mongodb_collection_name = os.getenv("MONGODB_COLLECTION_NAME")

# Connexion à la base de données MongoDB
cluster = MongoClient(mongodb_uri)
db = cluster[mongodb_db_name]
collection = db[mongodb_collection_name]

# Initialiser le pipeline NLP
nlp = pipeline("text-classification", model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")

def speech_to_text(audio_file_path):
    speech_config = speechsdk.SpeechConfig(subscription=subscription_key, region=region, speech_recognition_language="fr-FR")
    audio_config = speechsdk.audio.AudioConfig(filename=audio_file_path)
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    print("Reconnaissance vocale en cours à partir du fichier audio...")
    result = speech_recognizer.recognize_once()

    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print("Texte reconnu: {}".format(result.text))
        return result.text
    elif result.reason == speechsdk.ResultReason.NoMatch:
        print("Aucun texte reconnu: {}".format(result.no_match_details))
        return None
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print("Reconnaissance annulée: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print("Erreur: {}".format(cancellation_details.error_details))
        return None

def analyze_text(text):
    if text:
        sentiment_result = nlp(text)
        print("Résultat de l'analyse de sentiment: {}".format(sentiment_result))
        return sentiment_result
    else:
        return None

# Créer une chaîne séquentielle LangChain
class SpeechToTextAndNLPChain:
    def __init__(self):
        self.steps = [
            self.speech_to_text_step,
            self.analyze_text_step
        ]
    
    def speech_to_text_step(self, inputs):
        audio_file_path = inputs["audio_file_path"]
        transcription = speech_to_text(audio_file_path)
        return {"transcription": transcription}
    
    def analyze_text_step(self, inputs):
        transcription = inputs["transcription"]
        sentiment_analysis = analyze_text(transcription)
        return {"transcription": transcription, "sentiment_analysis": sentiment_analysis}
    
    def run(self, inputs):
        state = inputs
        for step in self.steps:
            state.update(step(state))
        return state

def process_audio_file(audio_file_path):
    # Initialiser la chaîne
    chain = SpeechToTextAndNLPChain()
    
    # Exécuter la chaîne
    result = chain.run({"audio_file_path": audio_file_path})
    
    if result and result.get("transcription") and result.get("sentiment_analysis"):
        # Enregistrer la transcription et l'analyse de sentiment dans la base de données
        transcription_data = {
            "audio_file_path": audio_file_path,
            "transcription": result["transcription"],
            "sentiment_analysis": result["sentiment_analysis"]
        }
        collection.insert_one(transcription_data)
        print("Transcription et analyse de sentiment enregistrées dans la base de données.")
    else:
        print("Échec du traitement de l'audio.")

if __name__ == "__main__":
    audio_file_path = "converted_1.wav"
    process_audio_file(audio_file_path)


