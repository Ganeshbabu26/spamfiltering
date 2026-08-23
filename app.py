
from fastapi import FastAPI, Request
from pydantic import BaseModel
import joblib

app = FastAPI(
    title="Spam Detection API",
    version="1.0"
)

# Load ML model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")


# Request JSON structure
class EmailRequest(BaseModel):
    message: str


# Response JSON structure
class SpamResponse(BaseModel):
    prediction: str
    confidence: float

@app.get("/health")
async def health_check():
    return {"status": "alive"}

@app.post("/predict", response_model=SpamResponse)
async def predict_spam(email: EmailRequest, request: Request):

    print("=================================")
    print("SPAM DETECTION REQUEST")
    print("=================================")

    print("REQUEST HEADERS:")
    print(request.headers)

    print("=================================")
    print("EMAIL MESSAGE:")
    print(email.message)

    print("=================================")

    # Convert email text into feature vector
    message_vectorized = vectorizer.transform(
        [email.message]
    )

    # Predict spam/ham
    prediction = model.predict(
        message_vectorized
    )[0]

    # Get probability
    probabilities = model.predict_proba(
        message_vectorized
    )[0]

    confidence = max(probabilities)

    print("PREDICTION:", prediction)
    print("CONFIDENCE:", confidence)

    print("=================================")

    return SpamResponse(
        prediction=str(prediction),
        confidence=round(float(confidence), 4)
    )