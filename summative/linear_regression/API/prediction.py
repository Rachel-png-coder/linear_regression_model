"""
Power Consumption Prediction API
FastAPI endpoint for predicting building energy usage
"""

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
import uvicorn
import os
from datetime import datetime
import io
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
    global model, scaler, features, last_training_time
    
    print("=" * 50)
    print("Loading Model Artifacts...")
    print("=" * 50)
    
    # Try multiple paths
    possible_paths = [
        "saved_model/Random Forest_model.pkl",
        "saved_model/best_model.pkl",
        "Random Forest_model.pkl",
        "best_model.pkl",
        "/app/saved_model/Random Forest_model.pkl",
    ]
    
    model_found = False
    for path in possible_paths:
        if os.path.exists(path):
            MODEL_PATH = path
            SCALER_PATH = path.replace("Random Forest_model.pkl", "scaler.pkl").replace("best_model.pkl", "scaler.pkl")
            FEATURES_PATH = path.replace("Random Forest_model.pkl", "feature_names.pkl").replace("best_model.pkl", "feature_names.pkl")
            print(f"✅ Found model at: {MODEL_PATH}")
            model_found = True
            break
    
    if not model_found:
        print("❌ Model file not found")
        return False
    
    try:
        model = joblib.load(MODEL_PATH)
        print(f"✅ Model loaded: {type(model).__name__}")
        
        if os.path.exists(SCALER_PATH):
            scaler = joblib.load(SCALER_PATH)
            print(f"✅ Scaler loaded")
        
        if os.path.exists(FEATURES_PATH):
            features = joblib.load(FEATURES_PATH)
            print(f"✅ Features loaded: {features}")
        
        last_training_time = datetime.now()
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False


# ============================================
# Date Parsing Function - FIXED FOR YOUR FORMAT
# ============================================

def parse_dates_flexible(df, date_column='DateTime'):
    """Parse dates with multiple possible formats"""
    
    # Make a copy to avoid warnings
    df = df.copy()
    
    # Convert to string and clean
    df[date_column] = df[date_column].astype(str).str.strip()
    
    print(f"📅 Sample dates: {df[date_column].head(3).tolist()}")
    
    # List of date formats to try (prioritizing YOUR format)
    date_formats = [
        '%m/%d/%Y %H:%M',      # 1/13/2017 0:00 - YOUR FORMAT
        '%m/%d/%Y %H:%M:%S',   # 1/13/2017 00:00:00
        '%m/%d/%y %H:%M',      # 1/13/17 0:00
        '%d/%m/%Y %H:%M',      # 13/1/2017 0:00
        '%Y-%m-%d %H:%M:%S',   # 2017-01-13 00:00:00
        '%Y-%m-%d %H:%M',      # 2017-01-13 00:00
        '%d-%m-%Y %H:%M',      # 13-01-2017 00:00
        '%m-%d-%Y %H:%M',      # 01-13-2017 00:00
    ]
    
    # Try each format
    parsed_success = False
    for date_format in date_formats:
        try:
            df[date_column] = pd.to_datetime(df[date_column], format=date_format, errors='raise')
            print(f"✅ Parsed dates with format: {date_format}")
            parsed_success = True
            break
        except (ValueError, TypeError):
            continue
    
    # If none worked, try with dayfirst parameter
    if not parsed_success:
        try:
            df[date_column] = pd.to_datetime(df[date_column], dayfirst=False, errors='raise')
            print("✅ Parsed with dayfirst=False (month/day format)")
            parsed_success = True
        except:
            pass
    
    # If still not working, try pandas auto-detection
    if not parsed_success:
        try:
            df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
            print("✅ Used pandas auto-detection")
            parsed_success = True
        except Exception as e:
            print(f"⚠️ Date parsing failed: {e}")
    
    # Check for failures
    if df[date_column].isna().sum() > 0:
        failed_count = df[date_column].isna().sum()
        print(f"⚠️ {failed_count} dates failed to parse")
        failed_examples = df[df[date_column].isna()][date_column].head(3).tolist()
        print(f"   Failed examples: {failed_examples}")
        raise ValueError(f"Unable to parse {failed_count} dates. Sample: {failed_examples}")
    
    print(f"✅ Successfully parsed {len(df)} dates")
    return df


# ============================================
# Helper Functions
# ============================================

def engineer_features(temperature, humidity, wind_speed, general_diffuse_flows, 
                      year, month, day, hour):
    """Create feature array with all required features"""
    from datetime import date
    
    try:
        day_of_week = date(year, month, day).weekday()
    except:
        day_of_week = 0
    
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


def retrain_model_with_new_data(new_data: pd.DataFrame):
    """Retrain model with new data"""
    global model, scaler, features, last_training_time, training_history
    
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_squared_error
    from sklearn.preprocessing import StandardScaler
    
    print("\n" + "="*50)
    print("RETRAINING MODEL")
    print("="*50)
    
    # ===== FIND AND PARSE DATE COLUMN =====
    date_col = None
    for col in new_data.columns:
        if 'date' in col.lower() or 'time' in col.lower() or 'DateTime' in col:
            date_col = col
            break
    
    if date_col:
        print(f"📅 Found date column: '{date_col}'")
        new_data = parse_dates_flexible(new_data, date_col)
        
        # Rename to DateTime if different
        if date_col != 'DateTime':
            new_data = new_data.rename(columns={date_col: 'DateTime'})
        
        # Extract features from DateTime
        new_data['Year'] = new_data['DateTime'].dt.year
        new_data['Month'] = new_data['DateTime'].dt.month
        new_data['Day'] = new_data['DateTime'].dt.day
        new_data['Hour'] = new_data['DateTime'].dt.hour
        new_data['DayOfWeek'] = new_data['DateTime'].dt.dayofweek
        
        print(f"📊 Date range: {new_data['DateTime'].min()} to {new_data['DateTime'].max()}")
    else:
        print("⚠️ No date column found, using default values")
        new_data['Year'] = 2017
        new_data['Month'] = 1
        new_data['Day'] = 1
        new_data['Hour'] = 12
        new_data['DayOfWeek'] = 0
    
    # ===== HANDLE COLUMN NAME VARIATIONS =====
    rename_map = {
        'Wind Speed': 'Wind_Speed',
        'general diffuse flows': 'general_diffuse_flows',
    }
    for old, new in rename_map.items():
        if old in new_data.columns and new not in new_data.columns:
            new_data = new_data.rename(columns={old: new})
    
    # ===== DEFINE FEATURES =====
    feature_cols = ['Temperature', 'Humidity', 'Wind_Speed', 
                    'general_diffuse_flows', 'Year', 'Month', 'Day', 
                    'Hour', 'DayOfWeek']
    
    # Check if all required columns exist
    missing_cols = [col for col in feature_cols if col not in new_data.columns]
    if missing_cols:
        print(f"❌ Missing columns: {missing_cols}")
        print(f"   Available columns: {list(new_data.columns)}")
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    print(f"✅ Using features: {feature_cols}")
    
    # ===== PREPARE DATA =====
    X = new_data[feature_cols].copy()
    
    # Find target column
    if 'Zone 1' in new_data.columns:
        y = new_data['Zone 1']
    elif 'Zone_1' in new_data.columns:
        y = new_data['Zone_1']
    else:
        raise ValueError("Target column 'Zone 1' not found")
    
    # Remove NaN
    initial_rows = len(X)
    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask]
    y = y[mask]
    
    print(f"📊 Data shape: {X.shape[0]} rows (removed {initial_rows - len(X)} rows with NaN)")
    
    if len(X) < 100:
        raise ValueError(f"Need at least 100 samples. Got {len(X)}")
    
    # ===== TRAIN MODEL =====
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale
    new_scaler = StandardScaler()
    X_train_scaled = new_scaler.fit_transform(X_train)
    X_test_scaled = new_scaler.transform(X_test)
    
    # Train
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
    
    print(f"✅ Model trained - R²: {r2:.4f}, RMSE: {rmse:.2f}")
    
    # ===== SAVE MODEL =====
    training_history.append({
        "timestamp": datetime.now().isoformat(),
        "samples": len(X),
        "r2_score": r2,
        "rmse": rmse
    })
    
    os.makedirs("saved_model", exist_ok=True)
    joblib.dump(new_model, MODEL_PATH)
    joblib.dump(new_scaler, SCALER_PATH)
    joblib.dump(feature_cols, FEATURES_PATH)
    
    # Update globals
    model = new_model
    scaler = new_scaler
    features = feature_cols
    last_training_time = datetime.now()
    
    print("="*50)
    print("✅ RETRAINING COMPLETE")
    print("="*50)
    
    return new_model, new_scaler, r2, len(X)


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
    
    @field_validator('temperature')
    @classmethod
    def validate_temperature(cls, v):
        if v < -20 or v > 50:
            raise ValueError(f'Temperature {v}°C is outside realistic range')
        return v
    
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
    predicted_power_kw: float
    timestamp: str
    model_used: str


class RetrainResponse(BaseModel):
    status: str
    message: str
    new_r2_score: Optional[float] = None
    samples_used: int
    timestamp: str


class TrainingStatusResponse(BaseModel):
    last_training_time: Optional[str]
    model_loaded: bool
    features_count: int
    training_history: list


# ============================================
# Endpoints
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


@app.get("/training/status")
async def training_status():
    """Get training status and history"""
    return {
        "last_training_time": last_training_time.isoformat() if last_training_time else None,
        "model_loaded": model is not None,
        "features_count": len(features) if features else 0,
        "training_history": training_history[-5:] if training_history else []
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make a power consumption prediction"""
    
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        features_array = engineer_features(
            request.temperature, request.humidity, request.wind_speed,
            request.general_diffuse_flows, request.year, request.month,
            request.day, request.hour
        )
        
        features_scaled = scaler.transform(features_array)
        prediction = model.predict(features_scaled)[0]
        
        return PredictionResponse(
            predicted_power_kw=float(prediction),
            timestamp=datetime.now().isoformat(),
            model_used="RandomForestRegressor"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/retrain", response_model=RetrainResponse)
async def retrain_model(file: UploadFile = File(...)):
    """Retrain model with new CSV data"""
    
    try:
        # Read uploaded file
        contents = await file.read()
        new_data = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        print(f"📊 Received {len(new_data)} rows")
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
        
        print(f"🧹 Cleaned: {cleaned_rows} rows (removed {initial_rows - cleaned_rows})")
        
        if cleaned_rows < 100:
            raise HTTPException(
                status_code=400,
                detail=f"Need at least 100 samples. Got {cleaned_rows}"
            )
        
        # Retrain
        _, _, r2, samples_used = retrain_model_with_new_data(new_data)
        
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
        raise HTTPException(status_code=400, detail=f"Data error: {str(e)}")
    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=f"Retraining error: {str(e)}")


# ============================================
# Run the API
# ============================================

# Load model on startup
load_model_artifacts()

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)