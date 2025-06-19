import os
from app import app

if __name__ == "__main__":
    # Get port from environment variable (Heroku sets this)
    port = int(os.environ.get("PORT", 5000))
    # Run the app - Heroku will use gunicorn instead of this
    app.run(host="0.0.0.0", port=port) 