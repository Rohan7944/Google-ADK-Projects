from client import UserApiClient

client = UserApiClient(
    base_url="http://127.0.0.1:9001",
    token="token-basic",
)

try:
    user = client.get_user()
    print(user)
except Exception as e:
    print(f"Errored: {e}")