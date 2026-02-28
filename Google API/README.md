# 📊 Google Sheets API Integration

A **FastAPI** application that integrates with the **Google Sheets API** via OAuth 2.0 to programmatically read and write data to Google Spreadsheets.

---

## ✨ Features

- **OAuth 2.0 Authentication** — Secure login flow via Google
- **Token Persistence** — Access tokens saved locally for reuse
- **Data Dump** — Programmatically write structured data to Google Sheets
- **RESTful API** — Clean FastAPI endpoints for login, callback, and data operations

---

## 🔄 Flow

```
1. User visits /login
2. Redirected to Google OAuth consent screen
3. Google calls back to /oauth/callback with auth code
4. Token saved locally
5. POST /dump-data writes data to the specified spreadsheet
```

---

## ▶️ How to Run

### 1. Prerequisites
- Python 3.8+
- A Google Cloud project with Sheets API enabled
- OAuth 2.0 credentials (`creds.json`) from Google Cloud Console

### 2. Install Dependencies
```bash
pip install fastapi uvicorn google-auth google-auth-oauthlib google-api-python-client
```

### 3. Setup
1. Place your `creds.json` in this folder
2. Update `SPREADSHEET_ID` in `main.py` with your Google Sheet ID

### 4. Start the Server
```bash
uvicorn main:app --reload --port 8000
```

### 5. Authenticate
Visit **http://localhost:8000/login** to authorize with Google.

### 6. Dump Data
```bash
curl -X POST http://localhost:8000/dump-data
```

---

## ⚠️ Security Note
- `creds.json` and `token.json` are excluded from Git via `.gitignore`
- Never commit API credentials to version control

---

## 🧠 Tech Stack
- **Backend:** FastAPI, Uvicorn
- **API:** Google Sheets API v4
- **Auth:** OAuth 2.0 (google-auth-oauthlib)
