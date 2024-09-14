from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Load the trained model
model = joblib.load('spam_classifier_model.pkl')

app = FastAPI()

# Define the request body
class EmailRequest(BaseModel):
    text: str

# Endpoint to predict if an email is spam
@app.post("/predict/")
async def predict_spam(email: EmailRequest):
    prediction = model.predict([email.text])
    result = "spam" if prediction[0] == 1 else "not spam"
    return {"prediction": result}
