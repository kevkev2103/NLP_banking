# DEEP LEARNING (NLP) Bank Sentiment-Analysis POC
BRIEF: https://zippy-twig-11a.notion.site/Brief-pour-le-D-veloppement-d-un-Proof-of-Concept-en-NLP-a3189e01b21a4638baf6194e32927b34

## 1. Speech To Text
Model imported from Azure Speech.
https://azure.microsoft.com/en-us/products/ai-services/speech-to-text
https://github.com/Azure-Samples/cognitive-services-speech-sdk/tree/master/samples/python/console

## 2. Translation
Model imported from HuggingFace. Translation from french to english.
https://huggingface.co/Helsinki-NLP/opus-mt-fr-en
https://github.com/Azure-Samples/cognitive-services-speech-sdk/blob/master/samples/python/console/translation_sample.py

## 3. Sentiment-Analysis
Model imported from HuggingFace. Three labels classification: *positive*, *neutral*, *negative*.
https://huggingface.co/mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis

## 4. LangChain
Use of LangChain to Chain the different models together.
https://learn.deeplearning.ai/courses/langchain/lesson/4/chains