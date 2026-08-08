"""
JWT verification for the /ask endpoint.

Mirrors web/backend/auth_utils.py's get_current_user_email() so the
chatbot trusts the same tokens the Web team issues at /login, instead
of trusting a client-supplied user_id in the request body.
"""
import os

import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

JWT_ALGORITHM = "HS256"

bearer_scheme = HTTPBearer()


def get_current_user_email(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> str:
    """Verify the JWT token and return the email (user identity) it belongs to."""
    jwt_secret_key = os.environ.get("JWT_SECRET_KEY")
    credentials_error = HTTPException(
        status_code=401,
        detail="Could not validate credentials.",
    )
    if not jwt_secret_key:
        raise HTTPException(
            status_code=500,
            detail="Server is not configured with a JWT secret.",
        )

    token = credentials.credentials
    try:
        payload = jwt.decode(token, jwt_secret_key, algorithms=[JWT_ALGORITHM])
        email = payload.get("sub")
        if email is None:
            raise credentials_error
        return email
    except jwt.PyJWTError:
        raise credentials_error