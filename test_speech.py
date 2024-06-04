import azure.cognitiveservices.speech as speechsdk
import os
from dotenv import load_dotenv
from pymongo import MongoClient


# Charger les variables d'environnement à partir du fichier .env
load_dotenv()

# Récupérer les clés d'abonnement et la région à partir des variables d'environnement
subscription_key = os.getenv("SUBSCRIPTION_KEY")
region = os.getenv("REGION")

try:
    speech_config = speechsdk.SpeechConfig(subscription=subscription_key, region=region)
    print("Configuration réussie")
except Exception as e:
    print(f"Erreur de configuration:{e}")

def speech_to_text_from_file(audio_file_path):
    speech_config = speechsdk.SpeechConfig(subscription=subscription_key, region=region, speech_recognition_language="fr-Fr")
    audio_config = speechsdk.audio.AudioConfig(filename=audio_file_path)

    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    print("Reconnaissance vocale en cours à partir du fichier audio...")

    result = speech_recognizer.recognize_once()
    

    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print("Texte reconnu:{}".format(result.text))
    elif result.reason == speechsdk.Resultreason.NoMatch:
        print("Aucun texte reconnu: {}".format(result.no_match_details))
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print("Reconnaissance annulée:{}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.Cancellationreason.Error:
            print("Erreur: {}".format(cancellation_details.error_details))
        return None

if __name__ == "__main__":
    
    audio_file_path = "converted_2.wav"
    speech_to_text_from_file(audio_file_path)




# Configuration de MongoDB

def confiure_mongodb(uri, db_name):
    client = MongoClient(uri)
    db = client[db_name]
    collection = db['transcriptions']
    return collection

# Stocker la transcription dans MongoDB

