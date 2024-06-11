import os
from dotenv import load_dotenv
import azure.cognitiveservices.speech as speechsdk
from pymongo import MongoClient
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
from transformers import pipeline
import openai
import pyaudio
import wave
import threading
import uuid
import subprocess

# From Mic
class Speech2TextModel:
    def __init__(self):
        load_dotenv()
        self.speech_key = os.getenv("speech_key")
        self.region = os.getenv("region")
        self.recording = False

    @staticmethod
    def generate_unique_filename():
        unique_id = uuid.uuid4()
        return str(unique_id)
    
    def _check_mic_available(self):
        p = pyaudio.PyAudio()
        num_devices = p.get_host_api_info_by_index(0).get('deviceCount')
        mic_available = False
        for i in range(num_devices):
            device_info = p.get_device_info_by_host_api_device_index(0, i)
            if device_info.get('maxInputChannels') > 0:
                print(f"Input Device id {i} - {device_info.get('name')}")
                mic_available = True
        p.terminate()
        return mic_available

    def is_mic_available(self):
        return self._check_mic_available()

    def record_audio(self, output_file):
        chunk = 1024
        sample_format = pyaudio.paInt16
        channels = 1
        fs = 44100  # samples per second
        p = pyaudio.PyAudio()  # creates an interface to PortAudio
        
        # Vérification des périphériques audio disponibles
        if not self._check_mic_available():
            print("Aucun microphone disponible. Tentative d'activation via PulseAudio...")
            try:
                subprocess.run(["pactl", "load-module", "module-loopback"], check=True)
            except subprocess.CalledProcessError as e:
                print(f"Erreur lors de l'activation du micro: {e}")
                raise Exception("Aucun microphone disponible après tentative d'activation.")
                return  # Arrêtez l'enregistrement si le micro ne peut pas être activé
            
        stream = p.open(format=sample_format,
                        channels=channels,
                        rate=fs,
                        frames_per_buffer=chunk,
                        input=True)
        frames = []  # initialize array to store frames
        print("Recording audio...")
        while self.recording:
            data = stream.read(chunk)
            frames.append(data)
        stream.stop_stream()
        stream.close()
        p.terminate()
        print("Saving audio...")
        wf = wave.open(output_file, "wb")
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(sample_format))
        wf.setframerate(fs)
        wf.writeframes(b''.join(frames))
        wf.close()

    def speak_to_microphone(self, stop_phrase, output_file):
        speech_config = speechsdk.SpeechConfig(subscription=self.speech_key, region=self.region)
        speech_config.speech_recognition_language = "fr-FR"
        speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config)

        print('Speak into your microphone. Say "stop session" to end the recording session.')

        self.recording = True

        # Generate unique filenames and save them in the corresponding folders
        base_filename = self.generate_unique_filename()
        audios_folder = "audios/"
        transcripts_folder = "transcripts/"
        # Ensure the directories exist
        os.makedirs(audios_folder, exist_ok=True)
        os.makedirs(transcripts_folder, exist_ok=True)
        audio_output_file = os.path.join(audios_folder, f"{base_filename}.wav")
        text_output_file = os.path.join(transcripts_folder, f"{base_filename}.txt")

        # Start audio recording in a separate thread
        audio_thread = threading.Thread(target=self.record_audio, args=(audio_output_file,))
        audio_thread.start()

        '''Speech to Text: Transcript'''
        with open(text_output_file, "w", encoding="utf-8") as file:
            while True:
                # Wait for any audio signal
                speech = speech_recognizer.recognize_once_async().get()

                if speech.reason == speechsdk.ResultReason.RecognizedSpeech:
                    # Generate unique filename for text_transcription
                    print('Recognized: {}'.format(speech.text))
                    file.write(speech.text + "\n")
                    if stop_phrase.lower() in speech.text.lower():
                        print("Session ended by user.")
                        self.recording = False  # stop recording
                        audio_thread.join()  # wait for the audio thread to finish
                        break

                # Check if the mic-speech is recognized, if not, print an error message (including logs/details).
                elif speech.reason == speechsdk.ResultReason.NoMatch:
                    print("No speech could be recognized: {}".format(speech.no_match_details))

                # Print an error message with details if transcript gets canceled for any reason
                elif speech.reason == speechsdk.ResultReason.Canceled:
                    cancellation_details = speech.cancellation_details
                    print("Speech Recognition canceled: {}".format(cancellation_details.reason))
                    if cancellation_details.reason == speechsdk.CancellationReason.Error:
                        print('Error details: {}'.format(cancellation_details.error_details))
                        print('Did you set the speech resource key and region values?')

        # Return the transcription content
        with open(text_output_file, "r", encoding="utf-8") as file:
            transcription = file.read()
        return transcription


class SentimentAnalysisModel:
    def __init__(self):
        load_dotenv()        
        self.azure_text_analytics_key = os.getenv("AZURE_TEXT_ANALYTICS_KEY")
        self.azure_text_analytics_endpoint = os.getenv("AZURE_TEXT_ANALYTICS_ENDPOINT")
        # Configuration du client Azure Text Analytics
        self.text_analytics_client = TextAnalyticsClient(
            endpoint=self.azure_text_analytics_endpoint,
            credential=AzureKeyCredential(self.azure_text_analytics_key)
)
    def analyze_sentiment(self, text):
        # a méthode analyze_sentiment du client Azure Text Analytics attend une liste de documents comme entrée
        result = self.text_analytics_client.analyze_sentiment(text)[0]
        sentiment = result.sentiment
        confidence_scores = {
            "positive": result.confidence_scores.positive,
            "neutral": result.confidence_scores.neutral,
            "negative": result.confidence_scores.negative
        }
        print(f"Sentiment Analysis: {result}")
        return {"sentiment": sentiment, "scores": confidence_scores}

class GenerateResponseModel:
    def __init__(self) -> None:
        load_dotenv()
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.api_base = os.getenv("OPENAI_API_BASE")
        self.api_deployment = os.getenv('OPENAI_DEPLOYMENT')
        self.api_version = os.getenv('OPENAI_VERSION')

    def generate_response(self, sentiment, text):
        prompt = f"Voici la transcription d'un message vocal laissé par un client auprès de sa banque. Analyse la transcription suivante et résume en quelques mots pourquoi le client est-il insatisfait. Transcription: {text}"
        # Appel à l'API OpenAI pour générer une réponse basée sur le modèle GPT-3.5-turbo.
        # Le modèle est configuré avec un ensemble de messages, dont un message système pour configurer le contexte et un message utilisateur contenant l'invite créée ci-dessus.
        response = openai.ChatCompletion.create(
            engine=self.api_deployment,  # Spécifie l'instance du modèle déployée à utiliser
            model="gpt-3.5-turbo",  
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},  # Message système pour définir le rôle de l'assistant
                {"role": "user", "content": prompt}  # Message utilisateur contenant l'invite
            ],
            max_tokens=100  # Limite le nombre de tokens dans la réponse générée à 100
        )
        # Extraction de la réponse générée via l'appel à l'API.
        llm_response = response['choices'][0]['message']['content'].strip()
        print("Réponse du modèle OpenAI : {}".format(llm_response))
        return llm_response