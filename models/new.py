import os
from dotenv import load_dotenv
import azure.cognitiveservices.speech as speechsdk
import pyaudio
import wave
import threading
import uuid

load_dotenv()
speech_key = os.getenv("speech_key")
region = os.getenv("region")

def generate_unique_filename():
    unique_id = uuid.uuid4()
    return str(unique_id)

def speak_to_microphone(speech_key, region, stop_phrase):

    '''Record Speech made with Mic and Save it as a .wav file'''
    def record_audio(output_file):
        global recording
        chunk = 1024
        sample_format = pyaudio.paInt16
        channels = 1
        fs = 44100 # samples per sec
        p = pyaudio.PyAudio() # creates an interface to PortAudio
        stream = p.open(format=sample_format,
                        channels=channels,
                        rate=fs,
                        frames_per_buffer=chunk,
                        input=True)
        frames = [] # initialize array to store frames
        print("Recording audio...")
        while True:
            data = stream.read(chunk)
            frames.append(data)
            # Break loop if recording is stopped
            if not recording:
                break
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

    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=region)
    speech_config.speech_recognition_language = "fr-FR"
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config)

    print('Speak into your microphone. Say "stop session" to end the recording session.')

    global recording
    recording = True

    # Generate unique filenames and save them in the corresponding folders
    base_filename = generate_unique_filename()
    audios_folder = "audios/"
    transcripts_folder = "transcripts/"
    # Ensure the directories exist
    os.makedirs(audios_folder, exist_ok=True)
    os.makedirs(transcripts_folder, exist_ok=True)
    audio_output_file = os.path.join(audios_folder, f"{base_filename}.wav")
    text_output_file = os.path.join(transcripts_folder, f"{base_filename}.txt")

    # Start audio recording in a separate thread
    audio_thread = threading.Thread(target=record_audio, args=(audio_output_file,))
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
                    recording = False # stop recording
                    audio_thread.join() # wait for the audio thread to finish
                    break

            # Check if the mic-speech is recognized, if not, print an error message (including logs/details).    
            elif speech.reason == speechsdk.ResultReason.NoMatch:
                print("No speech could be recognized: {}".format(speech.no_match_details))

            # Print an error message with details if transcript get canceled for any reason
            elif speech.reason == speechsdk.ResultReason.Canceled:
                cancellation_details = speech.cancellation_details
                print("Speech Recognition canceled: {}".format(cancellation_details.reason))
                if cancellation_details.reason == speechsdk.CancellationReason.Error:
                    print('Error details: {}'.format(cancellation_details.error_details))
                    print('Did you set the speech resource key and region values?')

# Initialize output path for transcription
stop_phrase = "stop session"
speak_to_microphone(speech_key=speech_key, region=region, stop_phrase=stop_phrase) 