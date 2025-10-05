# AI Disease Predictor from Symptoms

This project is a web-based application that uses a **deep learning model** to predict possible diseases based on symptoms provided by the user. It is built with a **Python backend using Flask** and a **React frontend**.

## Project Structure

```
AI Symptom Checker/
├── backend/
│ ├── app.py # Flask web server to handle predictions
│ ├── train_model.py # Script to train the prediction model
│ ├── models/ # Folder containing trained model files
│ ├── Symptom2Disease.csv # Dataset for training
│ └── requirements.txt # Python dependencies
├── frontend/ # React frontend
│ ├── package.json
│ └── src/ # React components, CSS, etc.
├── pipeline.sh # Script to install dependencies & run backend + frontend
└── README.md # This file
```

## Technologies Used

- **Backend:** Python, Flask  
- **Machine Learning:** TensorFlow/Keras, Scikit-learn, NLTK  
- **Frontend:** React, JavaScript, CSS 
- **Dataset**: [Symptom2Disease from Kaggle](https://www.kaggle.com/datasets/niyarrbarman/symptom2disease)

## Getting Started

Follow these instructions to set up and run the project on your local machine.

### Prerequisites

- Python 3.7+
- `pip` for Python package management  

### Installation

1. **Clone the repository (or download the project files)**
   ```bash
   git clone <your-repo-url>
   cd "AI Symptom Checker"
   ```
2. **Create and activate a virtual environment**
   It is highly recommended to use a virtual environment to keep the project's dependencies isolated.

   ```bash
   # Create a virtual environment named 'venv'
   python3 -m venv venv

   # Activate the virtual environment
   # On macOS/Linux:
   source venv/bin/activate
   # On Windows:
   # venv\Scripts\activate
   ```
3. **Install the required packages**
   Install all the necessary Python libraries from the `requirements.txt` file.

   ```bash
   pip install -r requirements.txt
   ```

   or run the following lines one by one
   ```bash
   pip install numpy pandas scikit-learn tensorflow
   pip install flask flask-cors
   pip install nltk
   ```
   After installing nltk run the following in python terminal(run 'python' in bash to access python terminal)
   ```bash
   import nltk
   nltk.download('stopwords')
   nltk.download('punkt')
   ```
   Exit python terminal with Ctrl+Z
   
4. **Install frontend dependencies**
   cd ../frontend
   npm install

## How to Run the Project

There are two main steps to run this application: training the model and then running the web server.

### Step 1: Train the Model

Before you can run the web application, you need to train the machine learning model. This script will process the data, train the model, and save the necessary files (`disease_predictor_model.h5`, `tfidf_vectorizer.pkl`, and `label_encoder.pkl`).

Run the following command in your terminal:

```bash
cd backend
python training_model.py
```

This will create the model files in the root directory of the project.

### Step 2: Run the Application

Once the model is trained and the files are saved, you can start the Flask server.
You can either run backend and frontend separately or use the pipeline.sh script to run both simultaneously.

#### Option A: Using pipeline.sh ####
   ```bash
   cd ..
   chmod +x pipeline.sh
   ./pipeline.sh
   ```
This will:
   - Install missing Python and Node dependencies
   - Train the model if not already trained
   - Start Flask backend (http://127.0.0.1:5000)
   - Start React frontend (http://localhost:3000)

#### Option B: Manually ####
##### Start Flask backend #####
   ```bash
   cd backend
   python app.py
   ```
##### Start React frontend #####
   ```bash
   cd ../frontend
   npm start
   ```
   The server will start.

### Step 3: Use the Application

Open your web browser and navigate to the following URL:

http://localhost:3000


You will see the "AI Disease Predictor" interface. Enter the symptoms in the text box or select from the dropdown and click the "Submit" button to get a prediction.

## Disclaimer

This tool is for informational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.
