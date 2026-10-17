from communityconnect import app

if __name__ == "__main__":
    if not app.secret_key:
        raise SystemExit("Set COMMUNITYCONNECT_SECRET_KEY in .env before starting the app.")
    app.run(debug=False)
