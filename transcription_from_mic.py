# YouTube tutorial
# https://youtu.be/2X5XBr19-G0?si=EC6aZicqn1bTgHhM

import os
from dotenv import load_dotenv
import azure.cognitiveservices.speech as speechsdk

load_dotenv()
speech_key = os.getenv("api_key")
service_region = os.getenv("region")

def speak_to_microphone(api_key, region, output_file):
    """performs one-shot speech recognition from the default microphone"""
    # <SpeechRecognitionWithMicrophone>
    speech_config = speechsdk.SpeechConfig(subscription=api_key, region=region)
    speech_config.speech_recognition_language = "fr-FR"
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config)
    # Creates a speech recognizer using microphone as audio input.
    # The default language is "en-us" -> changed to FR
    # audio_config = speechsdk.audio.AudioConfig()

    # Starts speech recognition, and returns after a single utterance is recognized. The end of a
    # single utterance is determined by listening for silence at the end or until a maximum of 15
    # seconds of audio is processed. It returns the recognition text as result.
    # Note: Since recognize_once() returns only a single utterance, it is suitable only for single
    # shot recognition like command or query.
    # For long-running multi-utterance recognition, use start_continuous_recognition() instead.
    result = speech_recognizer.recognize_once()

    # Set timeout durations
    speech_recognizer.properties.set_property(speechsdk.PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs, '60000')
    speech_recognizer.properties.set_property(speechsdk.PropertyId.SpeechServiceConnection_EndSilenceTimeoutMs, "20000")

    print('Speak into your microphone. Say "stop session" to end')

    with open(output_file, 'w', encoding='utf-8') as file:
        while True:
            speech_recognition_result = speech_recognizer.recognize_once_async().get() # wait for any audio signal

            if speech_recognition_result.reason == speechsdk.ResultReason.RecognizedSpeech: # check if any speech is detected
                recognized_text = speech_recognition_result.text
                print("Recognized: {}".format(recognized_text))
                file.write(recognized_text + "\n") # Write the recognized text to the file

                if "stop session" in speech_recognition_result.text.lower() or "stop session" in recognized_text.lower(): # end session
                    print("Session ended by user")
                    break

            elif speech_recognition_result.reason == speechsdk.ResultReason.NoMatch: # if audio not valid speech
                print("No speech could be recognized: {}".format(speech_recognition_result.no_match_details))
            elif speech_recognition_result.reason == speechsdk.ResultReason.Canceled: # if recording cancelled by user
                cancellation_details = speech_recognition_result.cancellation_details
                print("Speech Recognition canceled: {}".format(cancellation_details.reason))
                if cancellation_details.reason == speechsdk.CancellationReason.Error:
                    print("Error details: {}".format(cancellation_details.error_details))
                    print("Did you set the speech resource key and region values?")

output_file = "transcriptions/mic_transcript_avec_melody.txt"
speak_to_microphone(speech_key, service_region, output_file)