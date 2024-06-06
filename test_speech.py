import os
from dotenv import load_dotenv
from pymongo import MongoClient
import azure.cognitiveservices.speech as speechsdk
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential

# Charger les variables d'environnement à partir du fichier .env
load_dotenv()

# Récupérer les clés d'abonnement et la région à partir des variables d'environnement
subscription_key = os.getenv("SUBSCRIPTION_KEY")
region = os.getenv("REGION")
mongodb_uri = os.getenv("MONGODB_URI")
db_name = os.getenv("DB_NAME")
text_analytics_key = os.getenv("TEXT_ANALYTICS_KEY")
text_analytics_endpoint = os.getenv("TEXT_ANALYTICS_ENDPOINT")

if not all([subscription_key, region, mongodb_uri, db_name, text_analytics_key, text_analytics_endpoint]):
    raise ValueError("Toutes les variables d'environnement ne sont pas définies.")

# Configuration de MongoDB
def configure_mongodb(uri, db_name):
    client = MongoClient(uri, tls=True, tlsAllowInvalidCertificates=True)  # Désactive la vérification SSL
    db = client[db_name]
    collection = db['transcriptions']
    return collection

# Authentification du client Text Analytics
def authenticate_client(key, endpoint):
    ta_credential = AzureKeyCredential(key)
    text_analytics_client = TextAnalyticsClient(endpoint=endpoint, credential=ta_credential)
    return text_analytics_client

# Fonction pour convertir l'audio en texte
def speech_to_text_from_file(audio_file_path):
    try:
        speech_config = speechsdk.SpeechConfig(subscription=subscription_key, region=region, speech_recognition_language="fr-FR")
        audio_config = speechsdk.audio.AudioConfig(filename=audio_file_path)
        speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

        print("Reconnaissance vocale en cours à partir du fichier audio....")
        result = speech_recognizer.recognize_once()

        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            print("Texte reconnu: {}".format(result.text))
            return result.text
        elif result.reason == speechsdk.ResultReason.NoMatch:
            print("Aucun texte reconnu: {}".format(result.no_match_details))
        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            print("Reconnaissance annulée: {}".format(cancellation_details.reason))
            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                print("Erreur: {}".format(cancellation_details.error_details))
        return None
    except Exception as e:
        print(f"Erreur de reconnaissance vocale: {e}")
        return None

# Fonction pour analyser les sentiments
def analyze_sentiment(client, documents, language="fr"):
    try:
        response = client.analyze_sentiment(documents=documents, language=language)
        results = []
        for doc in response:
            if not doc.is_error:
                sentiment_score = doc.confidence_scores.positive - doc.confidence_scores.negative
                results.append({
                    "text": doc.sentences[0].text,
                    "sentiment": doc.sentiment,
                    "score": sentiment_score
                })
            else:
                results.append({
                    "text": doc.sentences[0].text if doc.sentences else "unknown",
                    "sentiment": "unknown",
                    "score": 0
                })
        return results
    except Exception as e:
        print(f"Erreur lors de l'analyse des sentiments: {e}")
        return []

# Stocker la transcription et les résultats d'analyse dans MongoDB
def store_transcription_and_analysis(collection, transcription, analysis_result):
    transcription_data = {
        "text": transcription,
        "sentiment": analysis_result["sentiment"],
        "score": analysis_result['score']
    }
    result = collection.insert_one(transcription_data)
    print(f"Transcription et analyse stockées avec l'ID: {result.inserted_id}")

if __name__ == "__main__":
    audio_file_path = "converted_2.wav"

    # Etape 1: Transcription de l'audio
    transcription = speech_to_text_from_file(audio_file_path)

    if transcription:
        # Etape 2 : analyse des sentiments
        text_analytics_client = authenticate_client(text_analytics_key, text_analytics_endpoint)
        analysis_results = analyze_sentiment(text_analytics_client, [transcription], language="fr")

        if analysis_results:
            analysis_result = analysis_results[0]

            # Etape 3: stocker la transcription et les résultats d'analyse dans MongoDB
            collection = configure_mongodb(mongodb_uri, db_name)
            store_transcription_and_analysis(collection, transcription, analysis_result)

        else:
            print("L'analyse des sentiments a échoué")

    else:
        print("La transcription a échoué")
