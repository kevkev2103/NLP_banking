# import os
# from dotenv import load_dotenv
# import azure.cognitiveservices.speech as speechsdk
# from pymongo import MongoClient
# from azure.ai.textanalytics import TextAnalyticsClient
# from azure.core.credentials import AzureKeyCredential
# import openai

# # Charger les variables d'environnement depuis le fichier .env
# load_dotenv()

# subscription_key = os.getenv("AZURE_SUBSCRIPTION_KEY")
# region = os.getenv("AZURE_REGION")
# mongodb_uri = os.getenv("MONGODB_URI")
# mongodb_db_name = os.getenv("MONGODB_DB_NAME")
# mongodb_collection_name = os.getenv("MONGODB_COLLECTION_NAME")
# text_analytics_key = os.getenv("AZURE_TEXT_ANALYTICS_KEY")
# text_analytics_endpoint = os.getenv("AZURE_TEXT_ANALYTICS_ENDPOINT")
# api_key = os.getenv("OPENAI_API_KEY")
# api_base = os.getenv("OPENAI_API_BASE")
# api_deployment = os.getenv('OPENAPI_DEPLOYMENT')
# api_version = os.getenv('OPENAPI_VERSION')

# # Connexion à la base de données MongoDB
# cluster = MongoClient(mongodb_uri)
# db = cluster[mongodb_db_name]
# collection = db[mongodb_collection_name]

# # Configuration du client Azure Text Analytics
# text_analytics_client = TextAnalyticsClient(
#     endpoint=text_analytics_endpoint,
#     credential=AzureKeyCredential(text_analytics_key)
# )

# # Configurer l'API OpenAI avec les informations d'Azure
# openai.api_key = api_key
# openai.api_base = api_base
# openai.api_type = 'azure'
# openai.api_version = api_version

# def speech_to_text(audio_file_path):
#     speech_config = speechsdk.SpeechConfig(subscription=subscription_key, region=region, speech_recognition_language="fr-FR")
#     audio_config = speechsdk.audio.AudioConfig(filename=audio_file_path)
#     speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

#     print("Reconnaissance vocale en cours à partir du fichier audio...")
#     result = speech_recognizer.recognize_once()

#     if result.reason == speechsdk.ResultReason.RecognizedSpeech:
#         print("Texte reconnu: {}".format(result.text))
#         return result.text
#     elif result.reason == speechsdk.ResultReason.NoMatch:
#         print("Aucun texte reconnu: {}".format(result.no_match_details))
#         return None
#     elif result.reason == speechsdk.ResultReason.Canceled:
#         cancellation_details = result.cancellation_details
#         print("Reconnaissance annulée: {}".format(cancellation_details.reason))
#         if cancellation_details.reason == speechsdk.CancellationReason.Error:
#             print("Erreur: {}".format(cancellation_details.error_details))
#         return None

# def analyze_text(text):
#     if text:
#         documents = [text]
#         response = text_analytics_client.analyze_sentiment(documents=documents)[0]
        
#         sentiment_result = {
#             "sentiment": response.sentiment,
#             "confidence_scores": {
#                 "positive": response.confidence_scores.positive,
#                 "neutral": response.confidence_scores.neutral,
#                 "negative": response.confidence_scores.negative
#             }
#         }
        
#         print("Résultat de l'analyse de sentiment: {}".format(sentiment_result))
#         return sentiment_result
#     else:
#         return None

# def analyze_with_llm(text):
#     prompt = f"Le client est insatisfait. Voici le texte: '{text}'. Pourquoi le client est-il insatisfait?"
#     response = openai.ChatCompletion.create(
#         engine=api_deployment,
#         model="gpt-3.5-turbo",
#         messages=[
#             {"role": "system", "content": "You are a helpful assistant."},
#             {"role": "user", "content": prompt}
#         ],
#         max_tokens=100  # Limiter le nombre de tokens à 100
#     )
#     llm_response = response['choices'][0]['message']['content'].strip()
#     print("Réponse du modèle OpenAI : {}".format(llm_response))
#     return llm_response

# # Créer une chaîne séquentielle LangChain
# class SpeechToTextAndNLPChain:
#     def __init__(self):
#         self.steps = [
#             self.speech_to_text_step,
#             self.analyze_text_step,
#             self.analyze_with_llm_step
#         ]
    
#     def speech_to_text_step(self, inputs):
#         audio_file_path = inputs["audio_file_path"]
#         transcription = speech_to_text(audio_file_path)
#         return {"transcription": transcription}
    
#     def analyze_text_step(self, inputs):
#         transcription = inputs["transcription"]
#         sentiment_analysis = analyze_text(transcription)
#         return {"transcription": transcription, "sentiment_analysis": sentiment_analysis}

#     def analyze_with_llm_step(self, inputs):
#         transcription = inputs["transcription"]
#         sentiment_analysis = inputs["sentiment_analysis"]

#         # Vérifier si le sentiment est négatif avant d'analyser avec le LLM
#         if sentiment_analysis["sentiment"] == "negative":
#             llm_analysis = analyze_with_llm(transcription)
#         else:
#             llm_analysis = None

#         return {"transcription": transcription, "sentiment_analysis": sentiment_analysis, "llm_analysis": llm_analysis}
    
#     def run(self, inputs):
#         state = inputs
#         for step in self.steps:
#             state.update(step(state))
#         return state

# def process_audio_file(audio_file_path):
#     # Initialiser la chaîne
#     chain = SpeechToTextAndNLPChain()
    
#     # Exécuter la chaîne
#     result = chain.run({"audio_file_path": audio_file_path})
    
#     if result and result.get("transcription") and result.get("sentiment_analysis"):
#         # Enregistrer la transcription et l'analyse de sentiment dans la base de données
#         transcription_data = {
#             "audio_file_path": audio_file_path,
#             "transcription": result["transcription"],
#             "sentiment_analysis": result["sentiment_analysis"],
#             "llm_analysis": result.get("llm_analysis")  # Peut être None si le sentiment n'est pas négatif
#         }
#         collection.insert_one(transcription_data)
#         print("Transcription, analyse de sentiment et analyse LLM (si applicable) enregistrées dans la base de données.")
#     else:
#         print("Échec du traitement de l'audio.")

# if __name__ == "__main__":
#     audio_file_path = "converted_7.wav"
#     process_audio_file(audio_file_path)

import os
from dotenv import load_dotenv
import azure.cognitiveservices.speech as speechsdk
from pymongo import MongoClient
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
import openai
import pickle

class Config:
    def __init__(self):
        load_dotenv()
        self.subscription_key = os.getenv("AZURE_SUBSCRIPTION_KEY")
        self.region = os.getenv("AZURE_REGION")
        self.mongodb_uri = os.getenv("MONGODB_URI")
        self.mongodb_db_name = os.getenv("MONGODB_DB_NAME")
        self.mongodb_collection_name = os.getenv("MONGODB_COLLECTION_NAME")
        self.text_analytics_key = os.getenv("AZURE_TEXT_ANALYTICS_KEY")
        self.text_analytics_endpoint = os.getenv("AZURE_TEXT_ANALYTICS_ENDPOINT")
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.api_base = os.getenv("OPENAI_API_BASE")
        self.api_deployment = os.getenv('OPENAPI_DEPLOYMENT')
        self.api_version = os.getenv('OPENAPI_VERSION')

config = Config()

# Connexion à la base de données MongoDB
cluster = MongoClient(config.mongodb_uri)
db = cluster[config.mongodb_db_name]
collection = db[config.mongodb_collection_name]

# Configuration du client Azure Text Analytics
text_analytics_client = TextAnalyticsClient(
    endpoint=config.text_analytics_endpoint,
    credential=AzureKeyCredential(config.text_analytics_key)
)

# Configurer l'API OpenAI avec les informations d'Azure
openai.api_key = config.api_key
openai.api_base = config.api_base
openai.api_type = 'azure'
openai.api_version = config.api_version

def speech_to_text(audio_file_path):
    speech_config = speechsdk.SpeechConfig(subscription=config.subscription_key, region=config.region, speech_recognition_language="fr-FR")
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
        documents = [text]
        response = text_analytics_client.analyze_sentiment(documents=documents)[0]
        
        sentiment_result = {
            "sentiment": response.sentiment,
            "confidence_scores": {
                "positive": response.confidence_scores.positive,
                "neutral": response.confidence_scores.neutral,
                "negative": response.confidence_scores.negative
            }
        }
        
        print("Résultat de l'analyse de sentiment: {}".format(sentiment_result))
        return sentiment_result
    else:
        return None

def analyze_with_llm(text):
    prompt = f"Le client est insatisfait. Voici le texte: '{text}'. Pourquoi le client est-il insatisfait?"
    response = openai.ChatCompletion.create(
        engine=config.api_deployment,
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=100  # Limiter le nombre de tokens à 100
    )
    llm_response = response['choices'][0]['message']['content'].strip()
    print("Réponse du modèle OpenAI : {}".format(llm_response))
    return llm_response

# Créer une chaîne séquentielle LangChain
class SpeechToTextAndNLPChain:
    def __init__(self):
        self.steps = [
            self.speech_to_text_step,
            self.analyze_text_step,
            self.analyze_with_llm_step
        ]
    
    def speech_to_text_step(self, inputs):
        audio_file_path = inputs["audio_file_path"]
        transcription = speech_to_text(audio_file_path)
        return {"transcription": transcription}
    
    def analyze_text_step(self, inputs):
        transcription = inputs["transcription"]
        sentiment_analysis = analyze_text(transcription)
        return {"transcription": transcription, "sentiment_analysis": sentiment_analysis}

    def analyze_with_llm_step(self, inputs):
        transcription = inputs["transcription"]
        sentiment_analysis = inputs["sentiment_analysis"]

        # Vérifier si le sentiment est négatif avant d'analyser avec le LLM
        if sentiment_analysis["sentiment"] == "negative":
            llm_analysis = analyze_with_llm(transcription)
        else:
            llm_analysis = None

        return {"transcription": transcription, "sentiment_analysis": sentiment_analysis, "llm_analysis": llm_analysis}
    
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
        # Préparer les données pour l'enregistrement
        transcription_data = {
            "audio_file_path": audio_file_path,
            "transcription": result["transcription"],
            "sentiment_analysis": result["sentiment_analysis"],
            "llm_analysis": result.get("llm_analysis")  # Peut être None si le sentiment n'est pas négatif
        }

        # Enregistrer les données dans un fichier pickle
        with open('transcription_data.pkl', 'wb') as f:
            pickle.dump(transcription_data, f)
        
        print("Transcription, analyse de sentiment et analyse LLM (si applicable) enregistrées dans un fichier pickle.")
    else:
        print("Échec du traitement de l'audio.")

if __name__ == "__main__":
    audio_file_path = "audios/converted_9.wav"
    process_audio_file(audio_file_path)