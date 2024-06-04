import os
from dotenv import load_dotenv
import time

load_dotenv()
speech_key = os.getenv("api_key")
service_region = os.getenv("region")

try:
    import azure.cognitiveservices.speech as speechsdk
except ImportError:
    print("""
    Importing the Speech SDK for Python failed.
    Refer to
    https://docs.microsoft.com/azure/cognitive-services/speech-service/quickstart-python for
    installation instructions.
    """)
    import sys
    sys.exit(1)

def transcription_from_audiofile(api_key, region, audio_file_path, output_file):
    """transcribes a conversation from an audio file (.wav)"""
    # Create speech config with subscription info
    speech_config = speechsdk.SpeechConfig(subscription=api_key, region=region)
    speech_config.speech_recognition_language = "fr-FR"
    
    # Create audio config from audio file
    audio_config = speechsdk.audio.AudioConfig(filename=audio_file_path)
    recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    print("Reconnaissance vocale en cours à partir du fichier audio...")

    result = recognizer.recognize_once()

    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print("Texte reconnu: {}".format(result.text))
        with open(output_file, "w", encoding='utf-8') as file:
            file.write(result.text)
    elif result.reason == speechsdk.ResultReason.NoMatch:
        print("Aucun texte reconnu: {}".format(result.no_match_details))
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print("La reconnaissance a été annulée: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print("Error details: {}".format(cancellation_details.error_details))

output_file = "transcriptions/audio_wav_transcript.txt"
transcription_from_audiofile(api_key=speech_key, region=service_region, audio_file_path="audio_files/converted_2.wav", output_file=output_file)