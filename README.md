# 🌿 AuraWellness – Agentic Student Wellness Platform

AuraWellness is an intelligent **AI-powered student wellness platform** that combines **Machine Learning** and **Agentic AI** to predict student stress levels and provide personalized mental health support.

The platform first analyzes a student's lifestyle using a trained Machine Learning model. When elevated stress levels are detected, it automatically hands over the interaction to an intelligent counseling agent built with **LangChain** and **Hugging Face**, enabling context-aware and empathetic conversations.

---

## 🚀 Features

### 🧠 Intelligent Stress Prediction
- Predicts student stress levels using lifestyle and environmental metrics.
- Uses a trained Scikit-Learn pipeline for real-time inference.
- Generates predictions within milliseconds.

### 🏆 Automated Model Selection
The training pipeline automatically trains and evaluates multiple machine learning models:

- Random Forest Classifier
- Logistic Regression
- Support Vector Machine (SVM)
- Gradient Boosting Classifier
- Decision Tree Classifier

The best-performing model is automatically selected and saved for production.

---

### 💬 Agentic AI Counseling

If a student is predicted to have **high stress**, AuraWellness automatically activates an AI counselor that:

- Maintains conversation history
- Understands the student's current health metrics
- Provides personalized wellness suggestions
- Responds empathetically using a fine-tuned Mental Health LLM

---

### 📊 Interactive Dashboard

The web interface includes:

- Lifestyle metric sliders
- Instant stress analysis
- Clean Tailwind CSS UI
- Real-time prediction
- AI chat interface

---

## 🛠 Tech Stack

### Backend

- FastAPI
- Python
- Uvicorn
- Pydantic

### Machine Learning

- Scikit-Learn
- Pandas
- NumPy
- Pickle

### Agentic AI

- LangChain
- Hugging Face Inference API
- Mental-Mistral-7B-Instruct

### Frontend

- HTML5
- Tailwind CSS
- Vanilla JavaScript
- Jinja2 Templates

---

# 📂 Project Structure

```text
aura-wellness-agent/
│
├── artifacts/
│   └── stress_pipeline_model.pkl
│
├── templates/
│   └── index.html
│
├── app.py
├── train.py
├── student-lifestyle-and-stress-dataset.csv
├── requirements.txt
├── .env
└── README.md
```

---

# ⚙ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/yourusername/aura-wellness-agent.git

cd aura-wellness-agent
```

---

## 2. Create Virtual Environment

### macOS / Linux

```bash
python3 -m venv venv

source venv/bin/activate
```

### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Configure Environment Variables

Create a `.env` file in the project root.

```env
HUGGINGFACEHUB_API_TOKEN=hf_your_token_here
```

---

## 5. Add Dataset

Place the dataset inside the project directory.

```
student-lifestyle-and-stress-dataset.csv
```

---

# ▶ Running the Project

## Step 1 — Train the Machine Learning Models

Run the automated ML tournament.

```bash
python train.py
```

This will:

- Load the dataset
- Perform preprocessing
- Train multiple ML models
- Compare their performance
- Save the best pipeline into

```
artifacts/stress_pipeline_model.pkl
```

---

## Step 2 — Start the FastAPI Server

```bash
python -m uvicorn app:app --reload
```

---

## Step 3 — Open the Dashboard

Visit

```
http://127.0.0.1:8000
```

---

# 🧠 Architecture

```text
                   Student Inputs
                         │
                         ▼
              Lifestyle Metrics (8 Features)
                         │
                         ▼
                FastAPI Backend API
                         │
                         ▼
           Serialized ML Pipeline (.pkl)
        ┌────────────────────────────────┐
        │                                │
        ▼                                ▼
   Low Stress                     High Stress
        │                                │
        ▼                                ▼
Healthy Feedback          Agentic AI Counselor
                                   │
                                   ▼
                           LangChain Memory
                                   │
                                   ▼
                  Hugging Face Mental-Mistral LLM
                                   │
                                   ▼
                  Personalized Counseling Response
```

---

# 🔄 Workflow

### 1. User Input

The student provides lifestyle information such as:

- Sleep Duration
- Study Hours
- Physical Activity
- Social Interaction
- Screen Time
- Diet Quality
- Stress Indicators
- Other wellness metrics

---

### 2. ML Prediction

The backend processes the data using:

- Ordinal Encoder
- Standard Scaler
- Best-performing ML classifier

The model predicts whether the student is experiencing:

- Low Stress
- High Stress

---

### 3. Intelligent Routing

### ✅ Low Stress

The dashboard displays a healthy wellness message.

### ⚠ High Stress

The application automatically activates the Agentic AI counselor.

---

### 4. AI Counseling

The counseling agent:

- Receives student metrics
- Creates a contextual prompt
- Maintains chat history
- Generates empathetic responses
- Continues conversation through the `/chat` endpoint

---

# 📌 API Endpoints

## Home

```
GET /
```

Returns the dashboard UI.

---

## Analyze Student Metrics

```
POST /analyze
```

Returns:

- Predicted stress level
- Confidence
- AI handoff (if required)

---

## Chat with AI Counselor

```
POST /chat
```

Maintains conversational memory and generates personalized wellness responses.

---

# 📈 Machine Learning Pipeline

The training pipeline includes:

- Data Cleaning
- Missing Value Handling
- Feature Encoding
- Standard Scaling
- Model Training
- Model Evaluation
- Automatic Best Model Selection
- Model Serialization

---

# 🤖 Agentic AI Pipeline

```text
Student Metrics
        │
        ▼
System Prompt
        │
        ▼
LangChain Memory
        │
        ▼
Mental-Mistral-7B-Instruct
        │
        ▼
Empathetic Personalized Response
```

---

# 🌟 Future Enhancements

- Multi-class stress prediction
- Anxiety and depression risk assessment
- Student wellness analytics dashboard
- PDF health reports
- Authentication and user profiles
- Therapist recommendation system
- Voice-based AI counseling
- Emotion detection from speech
- Mobile application
- Wearable device integration
- Real-time monitoring with IoT devices

---

# 📸 Dashboard Preview

> Add screenshots of your dashboard here.

```
images/
├── dashboard.png
├── prediction.png
└── chat.png
```

---

# 🤝 Contributing

Contributions are welcome.

1. Fork the repository

2. Create a feature branch

```bash
git checkout -b feature-name
```

3. Commit your changes

```bash
git commit -m "Add feature"
```

4. Push the branch

```bash
git push origin feature-name
```

5. Open a Pull Request.

---

# 📄 License

This project is intended for educational and research purposes.

---

# 👨‍💻 Author

**Aditya Pandey**

**B.Tech Information Technology**

**Indian Institute of Information Technology (IIIT) Una**

AI • Machine Learning • Deep Learning • NLP • Generative AI • Agentic AI

---

## ⭐ If you found this project useful, consider giving it a Star!