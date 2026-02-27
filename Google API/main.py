from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
import os
import json

app = FastAPI()

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
CLIENT_SECRETS_FILE = "creds.json"
REDIRECT_URI = "http://localhost:8000/oauth/callback"
SPREADSHEET_ID = "YOUR_SHEET_ID_HERE"

# Allow HTTP for local dev
os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

# Store tokens (simple version)
TOKEN_FILE = "token.json"


@app.get("/login")
def login():
    flow = Flow.from_client_secrets_file(
        CLIENT_SECRETS_FILE,
        scopes=SCOPES,
        redirect_uri=REDIRECT_URI
    )

    auth_url, _ = flow.authorization_url(
        prompt="consent",
        access_type="offline"
    )

    return RedirectResponse(auth_url)


@app.get("/oauth/callback")
def oauth_callback(code: str):
    flow = Flow.from_client_secrets_file(
        CLIENT_SECRETS_FILE,
        scopes=SCOPES,
        redirect_uri=REDIRECT_URI
    )

    flow.fetch_token(code=code)
    creds = flow.credentials

    # Save token
    with open(TOKEN_FILE, "w") as token:
        token.write(creds.to_json())

    return {"message": "OAuth successful! You can now dump data."}


@app.post("/dump-data")
def dump_data():
    if not os.path.exists(TOKEN_FILE):
        return {"error": "Login first at /login"}

    with open(TOKEN_FILE, "r") as token:
        creds_info = json.load(token)

    from google.oauth2.credentials import Credentials
    creds = Credentials.from_authorized_user_info(creds_info, SCOPES)

    service = build("sheets", "v4", credentials=creds)
    sheet = service.spreadsheets()

    values = [
        ["Name", "Age", "City"],
        ["Hardik", 21, "Delhi"],
        ["Rahul", 22, "Mumbai"],
        ["Aman", 23, "Pune"]
    ]

    body = {"values": values}

    result = sheet.values().update(
        spreadsheetId=SPREADSHEET_ID,
        range="Sheet1!A1",
        valueInputOption="RAW",
        body=body
    ).execute()

    return {
        "status": "success",
        "updatedCells": result.get("updatedCells")
    }
