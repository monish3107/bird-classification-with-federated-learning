# Federated Learning Bird Classifier

## Overview
This project is a full-stack federated learning system for bird image classification using TensorFlow, Flower, and Flask. It includes a web UI for predictions, a federated server, and client scripts.

## Features
- Federated learning with Flower and TensorFlow
- Flask REST API for predictions and stats
- Web UI for image upload and results
- Docker and manual setup options

## Project Structure
- `app.py`: Flask API server
- `server.py`: Flower federated server
- `client.py`: Flower federated client
- `index.html`, `script.js`: Web UI
- `requirements.txt`: Python dependencies
- `datasets/`: Training and test data
- `saved_models/`: Trained models

## Setup

### 1. Clone the repository
```bash
git clone <repo-url>
cd federated
```

### 2. Manual Setup (Windows)
- Install Python 3.8+
- (Optional) Create a virtual environment:
  ```bash
  python -m venv venv
  venv\Scripts\activate
  ```
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```
- Copy `.env.example` to `.env` and adjust settings if needed.

### 3. Docker Setup
```bash
docker build -t federated-bird .
docker run -p 5000:5000 federated-bird
```

## Running the App

### Start the Federated Server
```bash
python server.py
```

### Start a Federated Client (in a new terminal)
```bash
python client.py --client_number=1
```

### Start the Flask API (for web UI)
```bash
python run_production.py
```

### Access the Web UI
Open `http://localhost:5000` in your browser.

## API Endpoints
- `POST /predict` — Upload an image for prediction
- `GET /stats` — Get prediction stats
- `GET /metrics` — Get model metrics
- `GET /health` — Health check

## Troubleshooting
- Ensure the model is trained and present in `saved_models/` before starting the API.
- For Windows, use Waitress or Gunicorn for production serving.
- Check logs for errors: `app.log`

## License
MIT 