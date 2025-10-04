#!/bin/bash

# Exit immediately if a command fails
set -e

echo "Starting AI Disease Symptom Checker pipeline..."

# --- Step 1: Backend setup ---
cd backend
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install numpy pandas scikit-learn tensorflow flask flask-cors nltk

# --- Step 2: Download required NLTK data ---
echo "Downloading NLTK data..."
python - <<END
import nltk
nltk.download('stopwords')
nltk.download('punkt')
END

# --- Step 3: Run the training script ---
echo "Running training_model.py to generate model files..."
python training_model.py

# --- Step 4: Start Flask backend ---
echo "Starting Flask backend on port 5000..."
python app.py &
FLASK_PID=$!

# --- Step 5: Frontend setup ---
cd ../frontend
echo "Installing frontend dependencies..."
npm install

# --- Step 6: Start React frontend ---
echo "Starting React frontend on port 3000..."
npm start

# --- Step 7: Cleanup ---
# When React frontend exits, kill Flask backend
echo "Stopping Flask backend..."
kill $FLASK_PID
echo "Pipeline terminated."
