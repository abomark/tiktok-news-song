"""
One-time TikTok OAuth setup script.
Run this once to get your TIKTOK_REFRESH_TOKEN and save it to .env

Usage:
    python tiktok_oauth_setup.py

Steps:
    1. Opens a local server on port 8080
    2. Prints an auth URL — open it in your browser
    3. Authorise the app on TikTok
    4. Script captures the callback, exchanges the code for tokens
    5. Prints the TIKTOK_REFRESH_TOKEN to add to your .env
"""

import asyncio
import os
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlencode, urlparse

import httpx
from dotenv import load_dotenv

load_dotenv()

CLIENT_KEY = os.environ.get("TIKTOK_CLIENT_KEY", "")
CLIENT_SECRET = os.environ.get("TIKTOK_CLIENT_SECRET", "")
REDIRECT_URI = "http://localhost:8080/callback"
SCOPES = "user.info.basic,video.publish,video.upload"

_auth_code: str | None = None


class CallbackHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        global _auth_code
        parsed = urlparse(self.path)
        if parsed.path == "/callback":
            params = parse_qs(parsed.query)
            if "code" in params:
                _auth_code = params["code"][0]
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"<h1>Success! You can close this tab.</h1><p>Return to the terminal.</p>")
                print(f"\n[oauth] Authorization code received.")
            else:
                error = params.get("error", ["unknown"])[0]
                self.send_response(400)
                self.end_headers()
                self.wfile.write(f"<h1>Error: {error}</h1>".encode())
                print(f"\n[oauth] Error: {error}")

    def log_message(self, format, *args):
        pass  # suppress server logs


def build_auth_url() -> str:
    params = {
        "client_key": CLIENT_KEY,
        "scope": SCOPES,
        "response_type": "code",
        "redirect_uri": REDIRECT_URI,
        "state": "tiktok_news_song",
    }
    return "https://www.tiktok.com/v2/auth/authorize/?" + urlencode(params)


async def exchange_code_for_tokens(code: str) -> dict:
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            "https://open.tiktokapis.com/v2/oauth/token/",
            data={
                "client_key": CLIENT_KEY,
                "client_secret": CLIENT_SECRET,
                "code": code,
                "grant_type": "authorization_code",
                "redirect_uri": REDIRECT_URI,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        resp.raise_for_status()
        return resp.json()


def main():
    if not CLIENT_KEY or not CLIENT_SECRET:
        print("ERROR: TIKTOK_CLIENT_KEY and TIKTOK_CLIENT_SECRET must be set in .env")
        sys.exit(1)

    auth_url = build_auth_url()
    print("\n" + "="*60)
    print("TikTok OAuth Setup")
    print("="*60)
    print(f"\n1. Open this URL in your browser:\n\n   {auth_url}\n")
    print("2. Log in to TikTok and authorise the app.")
    print("3. You will be redirected to localhost — the script captures it automatically.")
    print("\nStarting local callback server on http://localhost:8080 ...")

    webbrowser.open(auth_url)

    server = HTTPServer(("localhost", 8080), CallbackHandler)
    server.timeout = 1

    print("Waiting for callback (60s timeout)...")
    for _ in range(60):
        server.handle_request()
        if _auth_code:
            break
    server.server_close()

    if not _auth_code:
        print("ERROR: Timed out waiting for OAuth callback. Try again.")
        sys.exit(1)

    print("[oauth] Exchanging code for tokens...")
    tokens = asyncio.run(exchange_code_for_tokens(_auth_code))

    refresh_token = tokens.get("refresh_token", "")
    access_token = tokens.get("access_token", "")
    open_id = tokens.get("open_id", "")

    print("\n" + "="*60)
    print("SUCCESS! Add these to your .env file:")
    print("="*60)
    print(f"\nTIKTOK_REFRESH_TOKEN={refresh_token}")
    print(f"\n(open_id for reference: {open_id})")
    print(f"(access_token expires in: {tokens.get('expires_in')}s)")

    # Optionally auto-write to .env
    env_path = ".env"
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            content = f.read()
        if "TIKTOK_REFRESH_TOKEN=" in content:
            import re
            content = re.sub(r"TIKTOK_REFRESH_TOKEN=.*", f"TIKTOK_REFRESH_TOKEN={refresh_token}", content)
        else:
            content += f"\nTIKTOK_REFRESH_TOKEN={refresh_token}\n"
        with open(env_path, "w") as f:
            f.write(content)
        print(f"\n[oauth] TIKTOK_REFRESH_TOKEN has been written to {env_path}")


if __name__ == "__main__":
    main()
