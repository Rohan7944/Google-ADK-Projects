from fastapi import FastAPI, HTTPException
from typing import  Dict

app = FastAPI(title="Sample Weather API", version="1.0")

# -------------------------------------------------
# Sample Weather Data (Mock)
# -------------------------------------------------

weather_data: Dict[str, Dict] = {
    "delhi": {
        "temperature": 32,
        "forecast": [
            {"day": "Day 1", "temperature": 33},
            {"day": "Day 2", "temperature": 34},
            {"day": "Day 3", "temperature": 32},
            {"day": "Day 4", "temperature": 31},
            {"day": "Day 5", "temperature": 30},
        ],
    },
    "mumbai": {
        "temperature": 29,
        "forecast": [
            {"day": "Day 1", "temperature": 29},
            {"day": "Day 2", "temperature": 30},
            {"day": "Day 3", "temperature": 30},
            {"day": "Day 4", "temperature": 29},
            {"day": "Day 5", "temperature": 28},
        ],
    },
    "bengaluru": {
        "temperature": 26,
        "forecast": [
            {"day": "Day 1", "temperature": 26},
            {"day": "Day 2", "temperature": 27},
            {"day": "Day 3", "temperature": 27},
            {"day": "Day 4", "temperature": 26},
            {"day": "Day 5", "temperature": 25},
        ],
    },
}

# -------------------------------------------------
# API Endpoints
# -------------------------------------------------

@app.get("/temperature/{city}")
def get_temperature(city: str):
    city_key = city.lower()
    if city_key not in weather_data:
        raise HTTPException(status_code=404, detail="City not found")

    return {
        "city": city.title(),
        "temperature": weather_data[city_key]["temperature"],
        "unit": "°C",
    }


@app.get("/forecast/{city}")
def get_forecast(city: str):
    city_key = city.lower()
    if city_key not in weather_data:
        raise HTTPException(status_code=404, detail="City not found")

    return {
        "city": city.title(),
        "forecast_days": 5,
        "forecast": weather_data[city_key]["forecast"],
        "unit": "°C",
    }