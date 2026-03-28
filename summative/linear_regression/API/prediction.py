from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
import pickle
import numpy as np
import os
import joblib

app = FastAPI(title="Power Consumption Prediction API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the request model with ALL 9 features
class PredictionRequest(BaseModel):
    temperature: float = Field(..., ge=-50, le=60, description="Temperature in Celsius")
    humidity: float = Field(..., ge=0, le=100, description="Humidity percentage")
    wind_speed: float = Field(..., ge=0, le=200, description="Wind speed in km/h")
    general_diffuse_flows: float = Field(..., ge=0, le=2000, description="General diffuse flows in W/m²")
    year: int = Field(..., ge=2000, le=2030, description="Year")
    month: int = Field(..., ge=1, le=12, description="Month (1-12)")
    day: int = Field(..., ge=1, le=31, description="Day of month")
    hour: int = Field(..., ge=0, le=23, description="Hour of day (0-23)")
    day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0=Monday, 6=Sunday)")
    
    @field_validator('temperature')
    def validate_temperature(cls, v):
        if v < -50 or v > 60:
            raise ValueError('Temperature must be between -50 and 60')
        return v
    
    @field_validator('humidity')
    def validate_humidity(cls, v):
        if v < 0 or v > 100:
            raise ValueError('Humidity must be between 0 and 100')
        return v
    
    @field_validator('wind_speed')
    def validate_wind_speed(cls, v):
        if v < 0 or v > 200:
            raise ValueError('Wind speed must be between 0 and 200')
        return v
    
    @field_validator('general_diffuse_flows')
    def validate_general_diffuse_flows(cls, v):
        if v < 0 or v > 2000:
            raise ValueError('General diffuse flows must be between 0 and 2000')
        return v
    
    @field_validator('year')
    def validate_year(cls, v):
        if v < 2000 or v > 2030:
            raise ValueError('Year must be between 2000 and 2030')
        return v
    
    @field_validator('month')
    def validate_month(cls, v):
        if v < 1 or v > 12:
            raise ValueError('Month must be between 1 and 12')
        return v
    
    @field_validator('day')
    def validate_day(cls, v):
        if v < 1 or v > 31:
            raise ValueError('Day must be between 1 and 31')
        return v
    
    @field_validator('hour')
    def validate_hour(cls, v):
        if v < 0 or v > 23:
            raise ValueError('Hour must be between 0 and 23')
        return v
    
    @field_validator('day_of_week')
    def validate_day_of_week(cls, v):
        if v < 0 or v > 6:
            raise ValueError('Day of week must be between 0 and 6')
        return v

# Global variables
model = None
scaler = None
feature_names = None

def load_model():
    """Load the trained model and artifacts"""
    global model, scaler, feature_names
    
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        saved_model_dir = os.path.join(script_dir, 'saved_model')
        
        print("=" * 50)
        print("Loading Model Artifacts...")
        print("=" * 50)
        print(f"📁 Script location: {script_dir}")
        print(f"📁 Model path: {saved_model_dir}")
        
        if not os.path.exists(saved_model_dir):
            print(f"❌ saved_model directory not found")
            model = None
            return
        
        files = os.listdir(saved_model_dir)
        print(f"📁 Files in saved_model: {files}")
        
        # Load model
        model_path = os.path.join(saved_model_dir, 'Random Forest_model.pkl')
        
        if not os.path.exists(model_path):
            print(f"❌ Model file not found")
            model = None
            return
        
        print(f"✅ Found model at: {model_path}")
        
        # Load with joblib (since it worked before)
        try:
            model = joblib.load(model_path)
            print(f"✅ Model loaded successfully with joblib!")
            print(f"Model type: {type(model).__name__}")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            model = None
            return
        
        # Load feature names
        features_path = os.path.join(saved_model_dir, 'feature_names.pkl')
        if os.path.exists(features_path):
            try:
                with open(features_path, 'rb') as f:
                    feature_names = pickle.load(f)
                print(f"✅ Features loaded: {feature_names}")
                print(f"Expected number of features: {len(feature_names)}")
            except Exception as e:
                print(f"⚠️  Could not load feature names: {e}")
                feature_names = None
        
        # Load scaler if exists (ignore errors)
        scaler_path = os.path.join(saved_model_dir, 'scaler.pkl')
        if os.path.exists(scaler_path):
            try:
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
                print(f"✅ Scaler loaded")
            except Exception as e:
                print(f"⚠️  Could not load scaler: {e}")
                scaler = None
        
        print("=" * 50)
        print("✅ All artifacts loaded successfully!")
        print("=" * 50)
        
        # Test the model with dummy data (9 features)
        try:
            test_features = np.array([[25, 60, 10, 150, 2024, 6, 15, 14, 3]]).reshape(1, -1)
            test_pred = model.predict(test_features)
            print(f"✅ Model test successful! Test prediction: {test_pred[0]:.2f} kWh")
        except Exception as e:
            print(f"⚠️  Model test failed: {e}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        model = None

# Use lifespan event handler
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Starting up the API...")
    load_model()
    yield
    print("👋 Shutting down the API...")

# Create app with lifespan
app = FastAPI(title="Power Consumption Prediction API", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "message": "Power Consumption Prediction API",
        "status": "running",
        "model_loaded": model is not None,
        "model_type": type(model).__name__ if model else None,
        "features_expected": len(feature_names) if feature_names else 9,
        "feature_names": feature_names if feature_names else [
            "Temperature", "Humidity", "Wind_Speed", "general_diffuse_flows", 
            "Year", "Month", "Day", "Hour", "DayOfWeek"
        ],
        "endpoints": {
            "predict": "/predict (POST)",
            "docs": "/docs",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_type": type(model).__name__ if model else None,
        "features_expected": len(feature_names) if feature_names else 9
    }

@app.post("/predict")
async def predict(request: PredictionRequest):
    """
    Make a power consumption prediction based on input parameters
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check if model file exists"
        )
    
    try:
        # Prepare features in the correct order (9 features)
        # Order: Temperature, Humidity, Wind_Speed, general_diffuse_flows, 
        #        Year, Month, Day, Hour, DayOfWeek
        features = np.array([
            request.temperature,
            request.humidity,
            request.wind_speed,
            request.general_diffuse_flows,
            request.year,
            request.month,
            request.day,
            request.hour,
            request.day_of_week
        ]).reshape(1, -1)
        
        print(f"📊 Features shape: {features.shape}")
        print(f"📊 Features: {features}")
        
        # Apply scaler if available
        if scaler is not None:
            try:
                features = scaler.transform(features)
                print(f"📊 Scaled features: {features}")
            except Exception as e:
                print(f"⚠️  Scaler transform failed: {e}")
        
        # Make prediction
        prediction = model.predict(features)
        
        print(f"🎯 Prediction: {prediction[0]:.2f} kWh")
        
        # Return the prediction
        return {
            "prediction": float(prediction[0]),
            "input_features": request.model_dump(),
            "units": "kWh"
        }
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)