from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Dict, List, Optional
from pydantic import BaseModel

# -------------------------------------------------
# App
# -------------------------------------------------

app = FastAPI(
    title="Sample Weather API",
    version="1.0",
)

# -------------------------------------------------
# Security (Bearer Token)
# -------------------------------------------------

security = HTTPBearer(auto_error=False)

VALID_TOKEN = "my-secret-token"


def verify_bearer_token(
    credentials: HTTPAuthorizationCredentials = Security(security),
) -> str:
    if credentials is None:
        raise HTTPException(status_code=401, detail="Authorization header missing")

    if credentials.scheme != "Bearer":
        raise HTTPException(status_code=401, detail="Invalid auth scheme")

    if credentials.credentials != VALID_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    return credentials.credentials


# -------------------------------------------------
# Response models
# -------------------------------------------------

class ForecastDay(BaseModel):
    day: str
    temperature: int


class TemperatureResponse(BaseModel):
    city: str
    temperature: Optional[int] = None
    unit: Optional[str] = None
    message: Optional[str] = None


class ForecastResponse(BaseModel):
    city: str
    forecast_days: Optional[int] = None
    forecast: Optional[List[ForecastDay]] = None
    unit: Optional[str] = None
    message: Optional[str] = None


# -------------------------------------------------
# Mock Weather Data
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
# API Endpoints (Protected)
# -------------------------------------------------

@app.get("/temperature/{city}", response_model=TemperatureResponse)
def get_temperature(
    city: str,
    token: str = Depends(verify_bearer_token),
):
    city_key = city.lower()

    if city_key not in weather_data:
        return {
            "city": city.title(),
            "message": f"No temperature data available for '{city}'",
        }

    return {
        "city": city.title(),
        "temperature": weather_data[city_key]["temperature"],
        "unit": "°C",
    }


@app.get("/forecast/{city}", response_model=ForecastResponse)
def get_forecast(
    city: str,
    token: str = Depends(verify_bearer_token),
):
    city_key = city.lower()

    if city_key not in weather_data:
        return {
            "city": city.title(),
            "message": f"No forecast data available for '{city}'",
        }

    return {
        "city": city.title(),
        "forecast_days": 5,
        "forecast": weather_data[city_key]["forecast"],
        "unit": "°C",
    }
    
if __name__=="__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=9000,
        reload=True,
    )
