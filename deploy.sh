docker build -t melobgn/app_gradio_nlp .
docker push melobgn/app_gradio_nlp


az container create --resource-group RG_BUGNONM --file deploy.yaml