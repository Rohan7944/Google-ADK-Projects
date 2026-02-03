from .client import UserApiClient

def fetch_user(token: str):
    client = UserApiClient(
        base_url="http://127.0.0.1:9001",
        token=token,
    )

    try:
        user = client.get_user()
        if user:
            return user
    except Exception as e:
        print(f"Errored: {e}")
        return None