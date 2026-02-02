import requests
from typing import Dict, Any


class UserApiClient:
    def __init__(self, base_url: str, token: str):
        """
        :param base_url: API base URL (e.g. http://127.0.0.1:9001)
        :param token: Bearer token
        """
        self.base_url = base_url.rstrip("/")
        self.token = token

    def get_user(self) -> Dict[str, Any]:
        url = f"{self.base_url}/user"

        headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/json",
        }

        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code == 200:
            return response.json()

        # Handle known error cases cleanly
        try:
            error = response.json()
        except ValueError:
            error = {"message": "Unknown error"}

        raise RuntimeError(
            f"API call failed ({response.status_code}): {error}"
        )