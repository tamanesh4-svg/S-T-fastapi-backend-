
from pathlib import Path
import joblib

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager


# Model directory
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR

models = {}


# Load all models when FastAPI starts
@asynccontextmanager
async def lifespan(app: FastAPI):
    global models

    try:
        models["ca1"] = joblib.load(MODEL_DIR / "model_ca1.pkl")
        models["ca2"] = joblib.load(MODEL_DIR / "model_ca2.pkl")
        models["ca3"] = joblib.load(MODEL_DIR / "model_ca3.pkl")

        print("All three ML models loaded successfully!")

    except Exception as e:
        raise RuntimeError(f"Model loading failed: {e}")

    yield

    models.clear()


app = FastAPI(
    title="Student Performance Prediction API",
    description="Predicts expected total marks out of 100",
    version="1.0.0",
    lifespan=lifespan
)


# CORS (useful if React later calls FastAPI directly)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# Request structure
class PredictionRequest(BaseModel):
    stage: int = Field(ge=1, le=3)
    attendance: float = Field(ge=0, le=100)

    ca1: float | None = Field(default=None, ge=0, le=25)
    ca2: float | None = Field(default=None, ge=0, le=25)
    ca3: float | None = Field(default=None, ge=0, le=25)


@app.get("/")
def home():
    return {
        "message": "Student Performance Prediction API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "models_loaded": list(models.keys())
    }


@app.post("/predict")
def predict(request: PredictionRequest):

    try:
        # Select model and prepare features
        if request.stage == 1:

            if request.ca1 is None:
                raise HTTPException(400, "CA1 marks are required")

            model = models["ca1"]

            input_data = [[
                request.ca1,
                request.attendance
            ]]

            known_marks = request.ca1
            remaining_marks = 75

        elif request.stage == 2:

            if request.ca1 is None or request.ca2 is None:
                raise HTTPException(
                    400, "CA1 and CA2 marks are required"
                )

            model = models["ca2"]

            input_data = [[
                request.ca1,
                request.ca2,
                request.attendance
            ]]

            known_marks = request.ca1 + request.ca2
            remaining_marks = 50

        else:

            if (
                request.ca1 is None
                or request.ca2 is None
                or request.ca3 is None
            ):
                raise HTTPException(
                    400, "CA1, CA2 and CA3 marks are required"
                )

            model = models["ca3"]

            input_data = [[
                request.ca1,
                request.ca2,
                request.ca3,
                request.attendance
            ]]

            known_marks = (
                request.ca1 + request.ca2 + request.ca3
            )

            remaining_marks = 25

        # Generate prediction
        prediction = float(model.predict(input_data)[0])

        # Enforce possible total marks range
        prediction = max(
            known_marks,
            min(prediction, known_marks + remaining_marks)
        )

        return {
            "success": True,
            "stage": request.stage,
            "expected_total": round(prediction, 2),
            "maximum_marks": 100
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )
