# import gradio as gr
# import pickle
# from pymongo import MongoClient


# # Charger les configurations et fonctions du modèle (Assurez-vous que ce code est dans le même script ou dans un module importé)
# from model import Config, speech_to_text, analyze_text, analyze_with_llm, SpeechToTextAndNLPChain


# config = Config()


# cluster = MongoClient(config.mongodb_uri)
# db = cluster[config.mongodb_db_name]
# collection = db[config.mongodb_collection_name]

# # Définir la fonction pour enregistrer les données dans MongoDB
# def enregistrer_dans_mongodb(data):
#     collection.insert_one(data)
#     return "Données enregistrées dans MongoDB."

# # Charger le modèle depuis le fichier .pkl
# with open('transcription_data.pkl', 'rb') as f:
#     model = pickle.load(f)

# # Définir la fonction qui traite le fichier audio
# def process_audio_file(audio_file_path):
#     # Utiliser les fonctions de traitement pour analyser l'audio
#     transcription = speech_to_text(audio_file_path)
#     sentiment_analysis = analyze_text(transcription)
#     llm_analysis = analyze_with_llm(transcription) if sentiment_analysis and sentiment_analysis["sentiment"] == "negative" else None
    
#     # Préparer les résultats
#     result = {
#         "transcription": transcription,
#         "sentiment_analysis": sentiment_analysis,
#         "llm_analysis": llm_analysis
#     }
#     enregistrer_dans_mongodb(result)
#     return result

# # def custom_button():
# #     return gr.Button("Enregistrer dans MongoDB", enregistrer_dans_mongodb)

# # Créer l'interface Gradio avec le bouton personnalisé pour MongoDB
# interface = gr.Interface(
#     fn=process_audio_file,
#     inputs=gr.Audio(type="filepath"),
#     outputs=gr.JSON(),
#     title="Analyse de sentiments de clients dans le contexte bancaire",
#     description="Chargez un fichier audio à analyser.",
#     allow_flagging="never",
#     theme='abidlabs/dracula_test',
# )


# # Lancer l'application Gradio
# if __name__ == "__main__":
#     interface.launch()


import gradio as gr
import pickle
from pymongo import MongoClient


# Charger les configurations et fonctions du modèle (Assurez-vous que ce code est dans le même script ou dans un module importé)
from model import Config, speech_to_text, analyze_text, analyze_with_llm, SpeechToTextAndNLPChain

config = Config()

cluster = MongoClient(config.mongodb_uri)
db = cluster[config.mongodb_db_name]
collection = db[config.mongodb_collection_name]

# Définir la fonction pour enregistrer les données dans MongoDB
def enregistrer_dans_mongodb(data):
    collection.insert_one(data)
    return "Données enregistrées dans MongoDB."

# Charger le modèle depuis le fichier .pkl
with open('transcription_data.pkl', 'rb') as f:
    model = pickle.load(f)

# Définir la fonction qui traite le fichier audio
def process_audio_file(audio_file_path):
    # Utiliser les fonctions de traitement pour analyser l'audio
    transcription = speech_to_text(audio_file_path)
    sentiment_analysis = analyze_text(transcription)
    llm_analysis = analyze_with_llm(transcription) if sentiment_analysis and sentiment_analysis["sentiment"] == "negative" else None
    
    # Préparer les résultats
    result = {
        "transcription": transcription,
        "sentiment_analysis": sentiment_analysis,
        "llm_analysis": llm_analysis
    }
        # Enregistrer la transcription et l'analyse de sentiment dans la base de données
    transcription_data = {
        "audio_file_path": audio_file_path,
        "transcription": result["transcription"],
        "sentiment_analysis": result["sentiment_analysis"],
        "llm_analysis": result["llm_analysis"]
    }
    collection.insert_one(transcription_data)
    
    # Formater les résultats pour un affichage soigné
    formatted_result = f"""
    **Transcription:** {transcription}

    **Analyse de sentiment:**
    - Sentiment: {sentiment_analysis['sentiment']}
    - Positif: {sentiment_analysis['confidence_scores']['positive']}
    - Neutre: {sentiment_analysis['confidence_scores']['neutral']}
    - Negatif: {sentiment_analysis['confidence_scores']['negative']}

    **Pourquoi le client est-il insatisfait ?**
    {llm_analysis if llm_analysis else 'Le client ne semble pas insatisfait.'}
    """
    return formatted_result

def custom_button():
    return gr.Button("Enregistrer dans MongoDB", enregistrer_dans_mongodb)

# Créer l'interface Gradio avec le bouton personnalisé pour MongoDB
with gr.Blocks(theme="freddyaboulton/dracula_revamped") as interface:
    gr.Markdown("# Analyse de sentiments de clients dans le contexte bancaire")
    gr.Markdown("Chargez un fichier audio à analyser.")
    audio_input = gr.Audio(type="filepath", label="Fichier audio")
    result_output = gr.Markdown(label="Résultat")
    
    submit_button = gr.Button("Analyser")
    submit_button.click(process_audio_file, inputs=audio_input, outputs=result_output)
    


# Lancer l'application Gradio
if __name__ == "__main__":
    interface.launch()
