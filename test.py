import os
import openai
from dotenv import load_dotenv

# Charger les variables d'environnement à partir du fichier .env
load_dotenv()

# Récupérer les valeurs des variables d'environnement
api_key = os.getenv('OPENAI_API_KEY')
api_base = os.getenv('OPENAI_API_BASE')
api_deployment = os.getenv('OPENAPI_DEPLOYMENT')
api_version = os.getenv('OPENAPI_VERSION')

# Configurer l'API OpenAI avec les informations d'Azure
openai.api_key = api_key
openai.api_base = api_base
openai.api_type = 'azure'
openai.api_version = api_version

# Fonction pour appeler l'API et obtenir une réponse
def get_response(prompt):
    response = openai.ChatCompletion.create(
        engine=api_deployment,
        messages=[
            {"role": "system", "content": "You are an orc."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=100
    )
    return response.choices[0].message['content'].strip()

# Exemple d'utilisation de la fonction
prompt = "What is the capital of France?"
response = get_response(prompt)
print(response)