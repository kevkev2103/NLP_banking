from .speech2text import SpeechToTextModel
from pydantic import Field
from langchain.chains.base import Chain
from transformers import pipeline
import pymongo

# Second Model: Translation (Fr-En)
class TranslationModel:
    def __init__(self):
        self.translation_pipeline = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")

    # Translate Transcription from French to English
    def translate(self, text):
        result = self.translation_pipeline(text)
        return result[0]['translation_text'] # Comme le pipeline retourne une liste de dictionnaires, result[0] accède au premier dictionnaire de cette liste.

# Creation of the 1st LangChain Chain with both models: SpeechToText & Translation
class Speech2TextTranslationChain(Chain):
    stt_model: SpeechToTextModel = Field(...)
    translation_model = TranslationModel = Field(...)
    collection: pymongo.collection.Collection = Field(init=False)

    def __init__(self, stt_model, translation_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stt_model = stt_model
        self.translation_model = translation_model
        self.collection = self.stt_model.collection

    def _call(self, inputs):
        audio_file_path = inputs['audio_file_path']
        output_file = inputs.get('output_file', 'transcriptions/audio_wav_transcript_translate.txt')
        transcription = self.stt_model.transcribe(audio_file_path, output_file)
        translation = self.translation_model.translate(transcription)
        return {'transcription':transcription, 'translation':translation}
    
    @property # Ce décorateur est utilisé pour définir une méthode qui peut être accédée comme un attribut. En d'autres termes, il permet de définir des getters pour les attributs calculés.
    def input_keys(self): # Ces méthodes définissent les clés d'entrée et de sortie de votre chaîne de traitement. Elles spécifient les noms des paramètres que la chaîne attend en entrée et les noms des résultats que la chaîne produira en sortie.
        return ['audio_file_path'] # retourne une liste avec une seule chaîne de caractères 
        # Cela signifie que la chaîne attendra un dictionnaire d'entrée avec une clé nommée audio_file_path.
    
    @property
    def output_keys(self):
        return ['transcription', 'translation'] 
    
    '''
    Ces propriétés sont importantes pour définir de manière explicite l'interface de votre chaîne,
    c'est-à-dire ce qu'elle attend en entrée et ce qu'elle produit en sortie.
    Cela permet à LangChain de gérer les chaînes de traitement de manière cohérente et modulable.
    '''