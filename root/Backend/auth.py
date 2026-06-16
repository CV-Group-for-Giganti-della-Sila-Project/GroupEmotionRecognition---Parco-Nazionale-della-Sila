import os
import httpx
import jwt
import requests
from fastapi import HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt.exceptions import ExpiredSignatureError, InvalidTokenError
from dotenv import load_dotenv

load_dotenv()

COGNITO_REGION = os.getenv("COGNITO_REGION", "eu-west-1")
COGNITO_USER_POOL_ID = os.getenv("COGNITO_USER_POOL_ID", "")
COGNITO_CLIENT_ID = os.getenv("COGNITO_CLIENT_ID", "")

JWKS_URL = f"https://cognito-idp.{COGNITO_REGION}.amazonaws.com/{COGNITO_USER_POOL_ID}/.well-known/jwks.json"
COGNITO_ENDPOINT = "https://cognito-idp.eu-west-1.amazonaws.com/"

_jwks_cache: dict | None = None

bearer_scheme = HTTPBearer()


async def verify_app_token(credentials: HTTPAuthorizationCredentials = Security(bearer_scheme)) -> dict:
    token = credentials.credentials
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            response = await client.post(
                COGNITO_ENDPOINT,
                headers={
                    "Content-Type": "application/x-amz-json-1.1",
                    "X-Amz-Target": "AWSCognitoIdentityProviderService.GetUser",
                },
                json={"AccessToken": token},
            )
            if response.status_code != 200:
                raise HTTPException(status_code=401, detail="Invalid or expired token")
            return response.json()
        except httpx.RequestError:
            raise HTTPException(status_code=401, detail="Token validation failed")


def _get_jwks() -> dict:
    global _jwks_cache
    if _jwks_cache is None:
        response = requests.get(JWKS_URL, timeout=10)
        response.raise_for_status()
        _jwks_cache = response.json()
    return _jwks_cache


def _find_public_key(token: str):
    header = jwt.get_unverified_header(token)
    kid = header.get("kid")
    jwks = _get_jwks()
    for key in jwks.get("keys", []):
        if key.get("kid") == kid:
            return jwt.algorithms.RSAAlgorithm.from_jwk(key)
    return None


async def verify_node_token(credentials: HTTPAuthorizationCredentials = Security(bearer_scheme)) -> dict:
    token = credentials.credentials
    public_key = _find_public_key(token)
    if public_key is None:
        raise HTTPException(status_code=401, detail="Unable to find matching public key")

    expected_issuer = f"https://cognito-idp.{COGNITO_REGION}.amazonaws.com/{COGNITO_USER_POOL_ID}"

    try:
        payload = jwt.decode(
            token,
            public_key,
            algorithms=["RS256"],
            options={"verify_aud": False},
        )
    except ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

    if payload.get("iss") != expected_issuer:
        raise HTTPException(status_code=401, detail="Invalid token issuer")

    if payload.get("client_id") != COGNITO_CLIENT_ID:
        raise HTTPException(status_code=401, detail="Unauthorized client")

    return payload
