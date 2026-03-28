"""
Power Consumption Prediction API
FastAPI endpoint for predicting building energy usage
"""

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Optional
import uvicorn
import os
from datetime import datetime
import io
import tempfile
import warnings
warnings.filterwarnings('ignore')

# ============================================
# Initialize FastAPI App
# ============================================

app = FastAPI(
    title="Power Consumption Prediction API",
    description="Predict building power consumption using environmental factors",
    version="1.0.0"
)

# ============================================
# CORS Middleware
# ============================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# Load Model and Scaler at Startup
# ============================================

# Use environment variable for model path (for Render deployment)
MODEL_PATH = "saved_model/Random Forest_model.pkl"
SCALER_PATH = "saved_model/scaler.pkl"
FEATURES_PATH = "saved_model/feature_names.pkl"

# Global variables
model = None
scaler = None
features = None
last_training_time = None
training_history = []

def load_model_artifacts():
    """Load the trained model, scaler, and feature names"""
    global model, scaler, features, last_training_time, MODEL_PATH, SCALER_PATH, FEATURES_PATH
    
    print("=" * 50)
    print("Loading Model Artifacts...")
    print("=" * 50)
    print(f"Looking for model at: {MODEL_PATH}")
    
    # Check if the main path exists
    if not os.path.exists(MODEL_PATH):
        print(f"⚠️ Model not found at {MODEL_PATH}, trying alternative paths...")
        
        # Try alternative paths
        alt_paths = [
            "saved_model/Random Forest_model.pkl",
            "Random Forest_model.pkl",
            "/app/saved_model/Random Forest_model.pkl",
            os.path.join(os.path.dirname(__file__), "saved_model", "Random Forest_model.pkl")
        ]
        
        found = False
        for path in alt_paths:
            if os.path.exists(path):
                MODEL_PATH = path
                # Also update scaler and features paths
                SCALER_PATH = path.replace("Random Forest_model.pkl", "scaler.pkl")
                FEATURES_PATH = path.replace("Random Forest_model.pkl", "feature_names.pkl")
                print(f"✅ Found model at: {MODEL_PATH}")
                found = True
                break
        
        if not found:
            print("❌ Model file not found in any location")
            return False
    
    try:
        model = joblib.load(MODEL_PATH)
        print(f"✅ Model loaded: {type(model).__name__}")
        
        if os.path.exists(SCALER_PATH):
            scaler = joblib.load(SCALER_PATH)
            print(f"✅ Scaler loaded: {type(scaler).__name__}")
        else:
            print(f"⚠️ Scaler not found at {SCALER_PATH}")
            scaler = None
        
        if os.path.exists(FEATURES_PATH):
            features = joblib.load(FEATURES_PATH)
            print(f"✅ Features loaded: {len(features)} features")
            print(f"   Features: {features}")
        else:
            print(f"⚠️ Features not found at {FEATURES_PATH}")
            features = None
        
        last_training_time = datetime.now()
        print("\n✅ All artifacts loaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

# Load model on startup
load_success = load_model_artifacts()

# ============================================
# Pydantic Models
# ============================================

class PredictionRequest(BaseModel):
    """Request body for prediction endpoint"""
    
    temperature: float = Field(..., ge=-20, le=50, description="Temperature in Celsius")
    humidity: float = Field(..., ge=0, le=100, description="Humidity percentage")
    wind_speed: float = Field(..., ge=0, le=50, description="Wind speed in m/s")
    general_diffuse_flows: float = Field(..., ge=0, le=1000, description="General diffuse flows")
    year: int = Field(..., ge=2017, le=2025, description="Year")
    month: int = Field(..., ge=1, le=12, description="Month (1-12)")
    day: int = Field(..., ge=1, le=31, description="Day of month")
    hour: int = Field(..., ge=0, le=23, description="Hour of day (0-23)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "temperature": 25.0,
                "humidity": 60.0,
                "wind_speed": 2.5,
                "general_diffuse_flows": 100.0,
                "year": 2024,
                "month": 6,
                "day": 15,
                "hour": 14
            }
        }


class PredictionResponse(BaseModel):
    """Response body for prediction endpoint"""
    predicted_power_kw: float
    timestamp: str
    model_used: str


class RetrainResponse(BaseModel):
    """Response body for retrain endpoint"""
    status: str
    message: str
    new_r2_score: Optional[float] = None
    samples_used: int
    timestamp: str


class TrainingStatusResponse(BaseModel):
    """Response body for training status"""
    last_training_time: Optional[str]
    model_loaded: bool
    features_count: int
    training_history: list


# ============================================
# Helper Functions
# ============================================

def engineer_features(temperature, humidity, wind_speed, general_diffuse_flows, 
                      year, month, day, hour):
    """Create feature array with all required features"""
    from datetime import date
    day_of_week = date(year, month, day).weekday()
    
    features_array = np.array([[
        temperature,
        humidity,
        wind_speed,
        general_diffuse_flows,
        year,
        month,
        day,
        hour,
        day_of_week
    ]])
    
    return features_array


def parse_dates_flexible(df, date_column='DateTime'):
    """Parse dates with multiple possible formats"""
    
    # List of possible date formats
    date_formats = [
        '%m/%d/%Y %H:%M',      # 1/13/2017 0:00
        '%m/%d/%Y %H:%M:%S',   # 1/13/2017 00:00:00
        '%d/%m/%Y %H:%M',      # 13/1/2017 0:00
        '%Y-%m-%d %H:%M:%S',   # 2017-01-13 00:00:00
        '%Y-%m-%d %H:%M',      # 2017-01-13 00:00
        '%m-%d-%Y %H:%M',      # 01-13-2017 00:00
    ]
    
    for date_format in date_formats:
        try:
            df[date_column] = pd.to_datetime(df[date_column], format=date_format)
            print(f"✅ Successfully parsed dates with format: {date_format}")
            return df
        except (ValueError, TypeError):
            continue
    
    # If none work, try pandas auto-detection
    try:
        df[date_column] = pd.to_datetime(df[date_column], format='mixed')
        print("✅ Used pandas auto-detection for dates")
        return df
    except Exception as e:
        print(f"⚠️ Could not parse dates: {e}")
        raise ValueError(f"Unable to parse dates in column '{date_column}'. Please ensure dates are in a standard format.")


def retrain_model_with_new_data(new_data: pd.DataFrame):
    """Retrain model with new data"""
    global model, scaler, features, last_training_time, training_history
    
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_squared_error
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    
    # ===== FIX: Handle different date formats =====
    # Check if DateTime column exists and parse it
    if 'DateTime' in new_data.columns:
        new_data = parse_dates_flexible(new_data, 'DateTime')
    else:
        # Try to find a date-like column
        for col in new_data.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                new_data = parse_dates_flexible(new_data, col)
                break
    
    # Extract features from DateTime
    new_data['Year'] = new_data['DateTime'].dt.year
    new_data['Month'] = new_data['DateTime'].dt.month
    new_data['Day'] = new_data['DateTime'].dt.day
    new_data['Hour'] = new_data['DateTime'].dt.hour
    new_data['DayOfWeek'] = new_data['DateTime'].dt.dayofweek
    
    # Handle column name variations
    column_mapping = {
        'Wind Speed': 'Wind_Speed',
        'Wind_Speed': 'Wind_Speed',
        'general diffuse flows': 'general_diffuse_flows',
        'general_diffuse_flows': 'general_diffuse_flows',
    }
    
    for old_name, new_name in column_mapping.items():
        if old_name in new_data.columns and old_name != new_name:
            new_data = new_data.rename(columns={old_name: new_name})
    
    # Define features
    feature_cols = ['Temperature', 'Humidity', 'Wind_Speed', 
                    'general_diffuse_flows', 'Year', 'Month', 'Day', 
                    'Hour', 'DayOfWeek']
    
    # Check if all required columns exist
    missing_cols = [col for col in feature_cols if col not in new_data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Prepare features
    X = new_data[feature_cols]
    y = new_data['Zone 1']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    new_scaler = StandardScaler()
    X_train_scaled = new_scaler.fit_transform(X_train)
    X_test_scaled = new_scaler.transform(X_test)
    
    # Train new model
    new_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    new_model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = new_model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # Save training info
    training_info = {
        "timestamp": datetime.now().isoformat(),
        "samples": len(new_data),
        "r2_score": r2,
        "rmse": rmse
    }
    training_history.append(training_info)
    
    # Save new model files
    joblib.dump(new_model, MODEL_PATH)
    joblib.dump(new_scaler, SCALER_PATH)
    joblib.dump(feature_cols, FEATURES_PATH)
    
    # Update global variables
    model = new_model
    scaler = new_scaler
    features = feature_cols
    last_training_time = datetime.now()
    
    return new_model, new_scaler, r2, len(new_data)


# ============================================
# Health Check Endpoints
# ============================================

@app.get("/")
async def root():
    """Root endpoint - API health check"""
    return {
        "message": "Power Consumption Prediction API",
        "status": "running",
        "model_loaded": model is not None,
        "features": features,
        "last_training": last_training_time.isoformat() if last_training_time else None,
        "endpoints": {
            "predict": "/predict (POST)",
            "retrain": "/retrain (POST)",
            "training_status": "/training/status (GET)",
            "health": "/health (GET)",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if model is not None else "degraded",
        "model_available": model is not None,
        "scaler_available": scaler is not None,
        "last_training": last_training_time.isoformat() if last_training_time else None
    }


@app.get("/training/status", response_model=TrainingStatusResponse)
async def training_status():
    """Get training status and history"""
    return TrainingStatusResponse(
        last_training_time=last_training_time.isoformat() if last_training_time else None,
        model_loaded=model is not None,
        features_count=len(features) if features else 0,
        training_history=training_history[-5:]
    )


# ============================================
# Prediction Endpoint
# ============================================

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make a power consumption prediction"""
    
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Engineer features
        features_array = engineer_features(
            request.temperature,
            request.humidity,
            request.wind_speed,
            request.general_diffuse_flows,
            request.year,
            request.month,
            request.day,
            request.hour
        )
        
        # Scale features
        features_scaled = scaler.transform(features_array)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        
        return PredictionResponse(
            predicted_power_kw=float(prediction),
            timestamp=datetime.now().isoformat(),
            model_used="RandomForestRegressor"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


# ============================================
# Retrain Endpoint - Core Feature!
# ============================================

@app.post("/retrain", response_model=RetrainResponse)
async def retrain_model(
    file: UploadFile = File(..., description="CSV file with training data")
):
    """
    Retrain model with new CSV data
    
    Expected CSV columns:
    DateTime, Temperature, Humidity, Wind Speed, 
    general diffuse flows, diffuse flows, Zone 1, Zone 2, Zone 3
    """
    
    global model, scaler, features
    
    try:
        # Read uploaded file
        contents = await file.read()
        new_data = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        print(f"📊 Received data with {len(new_data)} rows")
        print(f"📋 Columns: {list(new_data.columns)}")
        
        # Validate required columns
        required_cols = ['Zone 1', 'Temperature', 'Humidity']
        missing_cols = [col for col in required_cols if col not in new_data.columns]
        
        if missing_cols:
            raise HTTPException(
                status_code=400, 
                detail=f"Missing required columns: {missing_cols}"
            )
        
        # Clean data
        initial_rows = len(new_data)
        new_data = new_data.dropna()
        cleaned_rows = len(new_data)
        
        print(f"🧹 Cleaned data: {cleaned_rows} rows (removed {initial_rows - cleaned_rows} rows with NaN)")
        
        if cleaned_rows < 100:
            raise HTTPException(
                status_code=400,
                detail=f"Need at least 100 samples. Got {cleaned_rows} after cleaning"
            )
        
        # Retrain model
        new_model, new_scaler, r2, samples_used = retrain_model_with_new_data(new_data)
        
        print(f"✅ Retraining complete! R² Score: {r2:.4f}")
        
        return RetrainResponse(
            status="success",
            message=f"Model retrained with {samples_used} samples",
            new_r2_score=r2,
            samples_used=samples_used,
            timestamp=datetime.now().isoformat()
        )
        
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Data validation error: {str(e)}")
    except Exception as e:
        print(f"❌ Retraining error: {e}")
        raise HTTPException(status_code=500, detail=f"Retraining error: {str(e)}")


# ============================================
# Stream Data Retraining Endpoint
# ============================================

@app.post("/retrain/stream")
async def retrain_from_stream(data: list[dict]):
    """
    Retrain model from streamed data (real-time)
    
    Send a list of data points to incrementally improve the model
    """
    
    try:
        if not data:
            raise HTTPException(status_code=400, detail="No data provided")
        
        # Convert to DataFrame
        new_data = pd.DataFrame(data)
        
        print(f"📊 Received {len(new_data)} streamed data points")
        
        # Validate required columns
        required_cols = ['Zone 1', 'Temperature', 'Humidity']
        missing_cols = [col for col in required_cols if col not in new_data.columns]
        
        if missing_cols:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns: {missing_cols}"
            )
        
        # Retrain model
        new_model, new_scaler, r2, samples_used = retrain_model_with_new_data(new_data)
        
        return {
            "status": "success",
            "message": f"Model retrained with {samples_used} streamed samples",
            "r2_score": r2,
            "samples_processed": samples_used,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stream retraining error: {str(e)}")


# ============================================
# Run the API
# ============================================

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)