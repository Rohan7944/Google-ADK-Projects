from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Set, Dict

# -------------------------------------------------
# App
# -------------------------------------------------

app = FastAPI(
    title="User Access API",
    version="1.0",
)

# -------------------------------------------------
# Sample Database
# -------------------------------------------------

USER_DB: Dict[str, Dict] = {
    "token-basic": { # Token to be compared
        "user_id": "user_basic", # User information passed
        "allowed_tags": {"basic"},
    },
    "token-premium": { # Token to be compared
        "user_id": "user_premium", # User information passed
        "allowed_tags": {"basic", "premium"},
    },
    "token-mid": { # Token to be compared
        "user_id": "user_mid", # User information passed
        "allowed_tags": set(),
    },
}

# -------------------------------------------------
# Security (Bearer Token)
# -------------------------------------------------

security = HTTPBearer(auto_error=False)


def verify_bearer_token(
    credentials: HTTPAuthorizationCredentials = Security(security),
) -> Dict:
    if credentials is None:
        raise HTTPException(
            status_code=401,
            detail="Authorization header missing",
        )

    if credentials.scheme != "Bearer":
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication scheme",
        )

    token = credentials.credentials

    if token not in USER_DB:
        raise HTTPException(
            status_code=401,
            detail="Invalid or unknown token",
        )

    return USER_DB[token]


# -------------------------------------------------
# Response Model
# -------------------------------------------------

class UserResponse(BaseModel):
    user_id: str
    allowed_tags: Set[str]


class ErrorResponse(BaseModel):
    message: str


# -------------------------------------------------
# API Endpoint
# -------------------------------------------------

@app.get(
    "/user",
    response_model=UserResponse,
    responses={401: {"model": ErrorResponse}},
)
def get_user_info(
    user_data: Dict = Depends(verify_bearer_token),
):
    return {
        "user_id": user_data["user_id"],
        "allowed_tags": user_data["allowed_tags"],
    }

if __name__=="__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=9001,
        reload=True,
    )