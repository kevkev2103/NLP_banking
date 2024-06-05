from .speech2text import SpeechToTextModel
from .translation import TranslationModel, Speech2TextTranslationChain
from .sentiment_analysis import SentimentAnalysisModel, SentimentAnalysisChain

# Utilisation du modèle avec LangChain
stt_model = SpeechToTextModel()
translation_model = TranslationModel()
collection = stt_model.collection  # Utilisation de la collection de SpeechToText

# Créez les chaînes
transcription_translation_chain = Speech2TextTranslationChain(stt_model=stt_model, translation_model=translation_model)
sentiment_model = SentimentAnalysisModel()
sentiment_analysis_chain = SentimentAnalysisChain(sentiment_model=sentiment_model, collection=collection)

# Première chaîne pour la transcription et la traduction
audio_file_path = 'audio_files/converted_2.wav'
result_translation = transcription_translation_chain({'audio_file_path': audio_file_path})

print("Transcription:", result_translation['transcription'])
print("Translation:", result_translation['translation'])

# Deuxième chaîne pour l'analyse de sentiment
result_sentiment = sentiment_analysis_chain({
    'transcription': result_translation['transcription'],
    'translation': result_translation['translation']
})

print("Sentiment:", result_sentiment['sentiment'])

# Pour run ce fichier, il faut effectuer ces commandes:
# cd /home/utilisateur/Documents/DEV-IA/projets/DEEP_LEARNING/NLP_banking
# python -m models.langchain