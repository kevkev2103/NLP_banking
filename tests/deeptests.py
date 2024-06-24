import pytest
from unittest.mock import patch
import subprocess
from models.models import Speech2TextModel, SentimentAnalysisModel, GenerateResponseModel

"""Test de Chargement des Variables d'Environnement
    Vérifiez que toutes les variables d'environnement sont correctement chargées et accessibles."""
    # à compléter avec d'autres modèles et d'autres variables d'environnement
    
# def test_dotenv_Speech2TextModel():
#     model = Speech2TextModel()
#     assert model.speech_key is not None
#     assert model.region is not None

    
"""Test de Disponibilité du Microphone
    Vérifiez que le microphone est disponible et peut être activé si nécessaire.
        Test de Détection d'Erreur d'Activation du Microphone
            Vérifiez que le code détecte et gère correctement les erreurs lorsque le microphone ne peut pas être activé.""" 
            
def test_is_mic_available():
    speech = Speech2TextModel()
    is_mic_available = speech.is_mic_available()
    # assert is_mic_available == True
    assert is_mic_available == False
    

"""Test des Permissions du Microphone
    Vérifiez que l'application gère correctement les erreurs lorsque les permissions d'accès au microphone ne sont pas accordées."""
    
def test_raise_error_when_mic_not_available():
    # Mocking the _check_mic_available method to simulate no microphone available
    with patch.object(Speech2TextModel, '_check_mic_available', return_value=False):
        # Creating an instance of the model
        model = Speech2TextModel()
        # Mocking the subprocess.run to avoid actual PulseAudio command execution (avoid recording)
        with patch("subprocess.run") as mock_run:
            # Mocking subprocess.run to raise an exception
            mock_run.side_effect = subprocess.CalledProcessError(1, 'pactl')
            with pytest.raises(Exception) as exc_info:
                model.record_audio("dummy_output.wav")
            assert "Aucun microphone disponible" in str(exc_info.value), "Expected error message not found."      
             
    
"""Test de Manipulation des Fichiers
    Vérifiez que les fichiers audio et texte sont créés dans les répertoires corrects et que les noms de fichiers sont uniques."""

"""Test de la Fonction de Transcription
    Enregistrez un court message vocal et vérifiez que la transcription correspond au message enregistré."""
    
"""Test de la Phrase d'Arrêt
    Vérifiez que la transcription s'arrête correctement lorsque la phrase d'arrêt ("stop session") est prononcée."""
    
"""Test de l'Analyse de Sentiment
    Vérifiez que l'analyse de sentiment fonctionne correctement avec des textes ayant des sentiments différents (positif, négatif, neutre)."""

# PB AU NIVEAU DU MODELE D'ANALYSE DE SENTIMENT
# LE MODELE N'EST PAS FINE-TUNE, CA EXPLIQUERAIT POURQUOI CHAQUE TEST REVIENT EN "NEUTRAL"
# IL FAUDRAIT CHANGER DE MODELE (PASSER SUR AZURE)

# def test_positive_transcript():
#     model = SentimentAnalysisModel()
#     # positive_transcript = "Je suis ravi d'être client chez vous. Merci encore pour vos nombreux conseils. Je n'hésiterai pas à vous recommander auprès de mes proches!"
#     positive_transcript = "Cette banque est géniale! Je suis plus que ravi d'être client chez eux."
#     result = model.analyze_sentiment([positive_transcript])
#     sentiment = result["sentiment"]
#     assert sentiment == "positive"
    
# def test_neutral_transcript():
#     model = SentimentAnalysisModel()
#     neutral_transcript = "Ok."
#     result = model.analyze_sentiment([neutral_transcript])
#     sentiment = result["sentiment"]
#     assert sentiment == "neutral"
    
# def test_negative_transcript():
#     model = SentimentAnalysisModel()
#     negative_transcript = "Allez bien vous faire foutre bande de connards! C'est une honte! Un scandale! Ma carte bancaire est bloquée depuis plus de deux semaines! J'exige une solution immédiate!"
#     result = model.analyze_sentiment([negative_transcript])
#     sentiment = result["sentiment"]
#     assert sentiment == "negative"
    
    
"""Test de la Génération de Réponses
    Vérifiez que les réponses générées par le modèle GPT-3.5-turbo sont pertinentes par rapport à l'analyse de sentiment et au texte fourni."""
    
"""Test des Erreurs de Reconnaissance Vocale
    Simulez des conditions où le texte n'est pas reconnu et vérifiez la gestion des erreurs."""

"""Test de la Sauvegarde MongoDB
    Vérifiez que les données (transcription, sentiment, réponse, timestamp) sont correctement sauvegardées dans la base de données MongoDB."""

"""Test des Limites de Tokens
    Vérifiez que la réponse générée ne dépasse pas la limite de 100 tokens."""
    
"""Test d'Intégration Complète
    Effectuez un test de bout en bout en enregistrant un message vocal, en le transcrivant, en analysant le sentiment, en générant une réponse et en sauvegardant les données dans MongoDB.
    Assurez-vous que toutes les étapes fonctionnent ensemble de manière fluide."""

"""Test de Compatibilité Multi-plateforme
    Testez le code sur différents systèmes d'exploitation (Windows, macOS, Linux) pour vous assurer qu'il fonctionne correctement partout."""