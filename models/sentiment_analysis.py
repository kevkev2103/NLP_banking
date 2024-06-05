from models.speech2text import SpeechToTextModel
from models.translation import TranslationModel
from pydantic import Field
from langchain.chains.base import Chain
from transformers import pipeline
import pymongo
from datetime import datetime

# Third model: Sentiment-Analysis (HuggingFace)
class SentimentAnalysisModel:
    def __init__(self):
        self.sentiment_pipeline = pipeline("text-classification", model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
    # Analyze sentiment from the translated transcription
    def analyze_sentiment(self, text):
        result = self.sentiment_pipeline(text)
        return result[0]

# Second LangChain chain with the first chain & Sentiment model
class SentimentAnalysisChain(Chain):
    sentiment_model: SentimentAnalysisModel = Field(...)
    # translation_model = TranslationModel = Field(...)
    collection: pymongo.collection.Collection = Field(init=False)

    def __init__(self, sentiment_model, collection, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.translation_model = translation_model
        self.sentiment_model = sentiment_model
        self.collection = collection

    def _call(self, inputs):
        transcription = inputs["transcription"]
        translation = inputs["translation"]
        sentiment = self.sentiment_model.analyze_sentiment(translation)
        timestamp = datetime.now()
        self.collection.insert_one({"transcription": transcription, "translation":translation, "timestamp": timestamp, "sentiment":sentiment})
        return {'transcription': transcription, 'translation':translation, 'sentiment':sentiment}
    
    @property
    def input_keys(self):
        return ['transcription', 'translation']
    
    @property
    def output_keys(self):
        return ['transcription', 'translation', 'sentiment']