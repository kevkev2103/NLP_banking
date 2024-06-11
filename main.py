import os
from dotenv import load_dotenv
import azure.cognitiveservices.speech as speechsdk
from pymongo import MongoClient, errors
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
import openai

#chargerment des variables d'env depuis le .env
load_dotenv()

#Récupération des variables d'env
subscription_key = os.getenv("AZURE_SUBSCRIPTION_KEY")
region = os.getenv("AZURE_REGION")
mongodb_uri = os.getenv("MONGODB_URI")
mongodb_db_name = os.getenv("MONGODB_DB_NAME")
mongodb_collection_name = os.getenv("MONGODB_COLLECTION_NAME")
text_analytics_key = os.getenv("AZURE_TEXT_ANALYTICS_KEY")
text_analytics_endpoint = os.getenv("AZURE_TEXT_ANALYTICS_ENDPOINT")
api_key = os.getenv("OPENAI_API_KEY")
api_base = os.getenv("OPENAI_API_BASE")
api_deployment = os.getenv('OPENAPI_DEPLOYMENT')
api_version = os.getenv('OPENAPI_VERSION')

# Configuration de l'API OpenAI
openai.api_key = api_key
openai.api_base = api_base
openai.api_type = 'azure'
openai.api_version = api_version

#connexion à la bdd MongoDB
try:
    cluster = MongoClient(mongodb_uri)
    db= cluster[mongodb_db_name]
    collection=db[mongodb_collection_name]
except errors.ConnectionError as e:
    print(f"Erreur de connexion à MongoDB: {e}")
    exit(1)

#Configuration du client text Analytics
text_analytics_client = TextAnalyticsClient(
    endpoint=text_analytics_endpoint,
    credential=AzureKeyCredential(text_analytics_key)
    )


def speech_to_text(audio_file_path):
    """Convertir un fichier audio en texte"""
    speech_config = speechsdk.SpeechConfig(subscription=subscription_key, region=region,speech_recognition_language="fr-FR")
    audio_config= speechsdk.audio.AudioConfig(filename=audio_file_path)
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config,audio_config= audio_config)

    print("Reconnaissance vocale en cours à partir du fichier audio..")
    result=speech_recognizer.recognize_once()

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

def store_transcription(audio_file_path, transcription):
    """Enregistrer la transcription dans MongoDB"""
    transcription_data = {
        "audio_file_path": audio_file_path,
        "transcription": transcription
    }
    try:
        collection.insert_one(transcription_data)
        print("Transcription enregistrée dans la base de données.")
    except errors.PyMongoError as e:
        print(f"Erreur lors de l'insertion dans MongoDB: {e}")

def process_audio_file(audio_file_path):
    """Traiter un fichier audio: conversion en texte et stockage"""
    transcription = speech_to_text(audio_file_path)
    if transcription:
        store_transcription(audio_file_path, transcription)
    else:
        print("Échec de la transcription.")

def analyze_sentiment(transcription):
    #Analyser le sentiment de la transcription
    documents=[transcription]
    response = text_analytics_client.analyze_sentiment(documents=documents)[0]
    sentiment_result = {
        "sentiment": response.sentiment,
        "confidence_scores":{
        "positive":response.confidence_scores.positive,
        "neutral":response.confidence_scores.neutral,
        "negative": response.confidence_scores.negative
        }
    }
    print("Resultat de l'analyse de sentiment:{}".format(sentiment_result))
    return sentiment_result

def store_sentiment_analysis(audio_file_path, sentiment_analysis):
    #Enregistrer l'analyse de sentiment dans MongoDB
    try:
        collection.update_one(
            {"audio_file_path": audio_file_path},
            {"$set":{"sentiment_analysis":sentiment_analysis}}
        )
        print("Analyse de sentiment enregistrée dans la base de données.")
    except errors.PyMongoerrror as e:
        print(f"erreur lors de la mise à jour dans MongoDB: {e}")


def process_sentiment_analysis():
    """Lire les transcriptions depuis MongoDB, analyser le sentiment et stocker les résultats"""
    try:
        transcriptions = collection.find({"transcription": {"$exists": True}, "sentiment_analysis": {"$exists": False}})
        for record in transcriptions:
            audio_file_path = record["audio_file_path"]
            transcription = record["transcription"]
            sentiment_analysis = analyze_sentiment(transcription)
            store_sentiment_analysis(audio_file_path, sentiment_analysis)
    except errors.PyMongoError as e:
        print(f"Erreur lors de la lecture depuis MongoDB: {e}")

def analyze_reason(transcription,sentiment):
    #Analyse de la raison du sentimet
    prompt= f"Le sentiment de la personne à l'origine de la transcription est {sentiment}.Voici la transcription:{transcription}.Pourquoi le sentiment de la personne est-il {sentiment}?"
    response = openai.ChatCompletion.create(
        engine=api_deployment,
        model="gpt-3.5-turbo",
        messages=[
            {"role":"system", "content":"You are a helpful assistant"},
            {"role":"user","content":prompt}
        ]
    )
    llm_response = response['choices'][0]['message']['content'].strip()
    print("Réponse du modèle OpenAI: {}".format(llm_response))
    return llm_response

def store_reason_analysis(audio_file_path, reason_analysis):
    #enregistrer l'analyse de la raison du sentiment dans MongoDB
    try:
        collection.update_one(
            {"audio_file_path": audio_file_path},
            {"$set":{"reason_analysis": reason_analysis}}
        )
        print("Analyse de la raison enregistrée dans la base de donnée.")
    except errors.PyMongoError as e:
        print(f"Erreur lors de la mise à jour dans MongoDB:{e}")

def process_reason_analysis():
    # lire les analyses de sentiment depuis MongoDB, analyser la raison et stocker les résultats
    try:
        sentiment_records = collection.find({"sentiment_analysis": {"$exists": True}, "reason_analysis":{"$exists": False}})
        for record in sentiment_records:
            audio_file_path = record["audio_file_path"]
            transcription = record["transcription"]
            sentiment = record["sentiment_analysis"]["sentiment"]
            reason_analysis = analyze_reason(transcription, sentiment)
            store_reason_analysis(audio_file_path, reason_analysis)
    except errors.PyMongoError as e:
        print(f"erreur lors de la lecture depuis MongoDB: {e}")



if __name__ == "__main__":
    audio_directory="audio"
    for filename in os.listdir(audio_directory):
        if filename.endswith(".wav"):
            audio_file_path=os.path.join(audio_directory,filename)
            process_audio_file(audio_file_path)
    process_sentiment_analysis()
    process_reason_analysis()
   