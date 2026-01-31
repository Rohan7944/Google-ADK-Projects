gcloud run deploy weather-mcp \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars \
    WEATHER_API_BASE_URL=https://your-weather-api.run.app,\
    BEARER_TOKEN=my-secret-token