import os
from dotenv import load_dotenv
import azure.cognitiveservices.speech as speechsdk
from pymongo import MongoClient

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

try:
    speech_config = speechsdk.SpeechConfig(subscription=subscription_key, region=region)
    print("Configuration réussie.")
except Exception as e:
    print(f"Erreur de configuration: {e}")

def speech_to_text_from_file(audio_file_path):
    speech_config = speechsdk.SpeechConfig(subscription=subscription_key, region=region, speech_recognition_language="fr-FR")
    audio_config = speechsdk.audio.AudioConfig(filename=audio_file_path)

    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    print("Reconnaissance vocale en cours à partir du fichier audio...")

    result = speech_recognizer.recognize_once()

    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print("Texte reconnu: {}".format(result.text))
        # Enregistrer la transcription dans la base de données
        transcription_data = {
            "audio_file_path": audio_file_path,
            "transcription": result.text
        }
        collection.insert_one(transcription_data)
        print("Transcription enregistrée dans la base de données.")
    elif result.reason == speechsdk.ResultReason.NoMatch:
        print("Aucun texte reconnu: {}".format(result.no_match_details))
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print("Reconnaissance annulée: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print("Erreur: {}".format(cancellation_details.error_details))

if __name__ == "__main__":
    audio_file_path = "converted_1.wav"
    speech_to_text_from_file(audio_file_path)

