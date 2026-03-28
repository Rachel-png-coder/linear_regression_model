"""
Power Consumption Prediction API
FastAPI endpoint for predicting building energy usage
Supports date format: DD-MM-YYYY HH:MM
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

# Get the directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))

# Set correct paths relative to this script
MODEL_PATH = os.path.join(current_dir, "saved_model", "Random Forest_model.pkl")
SCALER_PATH = os.path.join(current_dir, "saved_model", "scaler.pkl")
FEATURES_PATH = os.path.join(current_dir, "saved_model", "feature_names.pkl")

# Global variables
model = None
scaler = None
features = None
last_training_time = None
training_history = []


def parse_dates_flexible(df, date_column='DateTime'):
    """
    Parse dates with DD-MM-YYYY HH:MM format (primary)
    Also handles other common formats
    """
    
    print(f"🔍 Original date sample: {df[date_column].iloc[0] if len(df) > 0 else 'No data'}")
    
    # Step 1: Clean the data - remove extra spaces
    df[date_column] = df[date_column].astype(str).str.strip()
    
    # Step 2: Define formats specifically for your data
    date_formats = [
        # Your primary format: DD-MM-YYYY HH:MM
        '%d-%m-%Y %H:%M',        # 01-01-2017 00:00
        '%d-%m-%Y %H:%M:%S',     # 01-01-2017 00:00:00
        
        # Other common formats
        '%d/%m/%Y %H:%M',        # 01/01/2017 00:00
        '%d/%m/%Y %H:%M:%S',     # 01/01/2017 00:00:00
        '%Y-%m-%d %H:%M:%S',     # 2017-01-01 00:00:00
        '%Y-%m-%d %H:%M',        # 2017-01-01 00:00
        '%m/%d/%Y %H:%M',        # 01/13/2017 00:00
        '%m/%d/%Y %H:%M:%S',     # 01/13/2017 00:00:00
        
        # AM/PM formats (if needed)
        '%d-%m-%Y %I:%M %p',     # 01-01-2017 12:00 AM
        '%d/%m/%Y %I:%M %p',     # 01/01/2017 12:00 AM
        '%m/%d/%Y %I:%M %p',     # 01/13/2017 12:00 AM
    ]
    
    # Try each format
    for date_format in date_formats:
        try:
            df[date_column] = pd.to_datetime(df[date_column], format=date_format)
            print(f"✅ Successfully parsed with format: {date_format}")
            return df
        except (ValueError, TypeError):
            continue
    
    # Try pandas flexible parsing
    try:
        df[date_column] = pd.to_datetime(df[date_column])
        print(f"✅ Successfully parsed with pandas auto-detection")
        return df
    except Exception as e:
        print(f"⚠️ Date parsing failed: {e}")
        raise ValueError(f"Unable to parse dates. Sample values: {df[date_column].head(3).tolist()}")


def load_model_artifacts():
    """Load the trained model, scaler, and feature names"""
    global model, scaler, features, last_training_time
    
    print("=" * 50)
    print("Loading Model Artifacts...")
    print("=" * 50)
    
    print(f"📁 Script location: {current_dir}")
    print(f"📁 Model path: {MODEL_PATH}")
    print(f"📁 Model exists: {os.path.exists(MODEL_PATH)}")
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model file not found at: {MODEL_PATH}")
        
        # Check if saved_model exists in current directory
        if os.path.exists("saved_model"):
            print(f"✓ Found 'saved_model' in current directory")
            print(f"Contents: {os.listdir('saved_model')}")
            
            # Try to load using relative path
            try:
                model = joblib.load("saved_model/Random Forest_model.pkl")
                print(f"✅ Model loaded successfully from current directory!")
                print(f"Model type: {type(model).__name__}")
            except Exception as e:
                print(f"❌ Failed to load from current directory: {e}")
                return False
        else:
            print(f"✗ 'saved_model' not found in current directory")
            return False
    else:
        try:
            model = joblib.load(MODEL_PATH)
            print(f"✅ Model loaded successfully!")
            print(f"Model type: {type(model).__name__}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    # Load scaler
    if os.path.exists(SCALER_PATH):
        try:
            scaler = joblib.load(SCALER_PATH)
            print(f"✅ Scaler loaded")
        except Exception as e:
            print(f"⚠️ Could not load scaler: {e}")
    else:
        print(f"⚠️ Scaler not found at: {SCALER_PATH}")
    
    # Load features
    if os.path.exists(FEATURES_PATH):
        try:
            features = joblib.load(FEATURES_PATH)
            print(f"✅ Features loaded: {features[:5] if len(features) > 5 else features}")
        except Exception as e:
            print(f"⚠️ Could not load features: {e}")
    else:
        print(f"⚠️ Features file not found at: {FEATURES_PATH}")
    
    last_training_time = datetime.now()
    print("=" * 50)
    print("✅ All artifacts loaded successfully!")
    print("=" * 50)
    return True


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
            "retrain": "/retrain (POST) - Supports DD-MM-YYYY HH:MM format",
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
    """
    Retrain model with new CSV data
    Expected format: DD-MM-YYYY HH:MM (e.g., 01-01-2017 00:00)
    """
    
    try:
        # Read uploaded file
        contents = await file.read()
        
        # Try different encodings
        try:
            new_data = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        except UnicodeDecodeError:
            try:
                new_data = pd.read_csv(io.StringIO(contents.decode('latin-1')))
            except:
                new_data = pd.read_csv(io.StringIO(contents.decode('iso-8859-1')))
        
        print(f"\n{'='*50}")
        print(f"📊 Received file: {file.filename}")
        print(f"{'='*50}")
        print(f"Total rows: {len(new_data)}")
        
        # ===== FIX: Rename columns to match expected format =====
        print(f"\n📝 Normalizing column names...")
        
        # Rename columns to match expected names
        column_mapping = {
            'Wind Speed': 'Wind_Speed',
            'general diffuse flows': 'general_diffuse_flows',
            'Zone 1': 'Zone 1',
            'Temperature': 'Temperature',
            'Humidity': 'Humidity'
        }
        
        for old_name, new_name in column_mapping.items():
            if old_name in new_data.columns:
                new_data = new_data.rename(columns={old_name: new_name})
                print(f"   Renamed: '{old_name}' -> '{new_name}'")
        
        # ===== FIX: Parse dates with DD-MM-YYYY HH:MM format =====
        print(f"\n📅 Parsing dates...")
        
        if 'DateTime' in new_data.columns:
            # Parse dates (your format: 01-01-2017 00:00)
            new_data['DateTime'] = pd.to_datetime(new_data['DateTime'], format='%d-%m-%Y %H:%M', errors='coerce')
            
            # Check if parsing failed for some rows
            failed_parses = new_data['DateTime'].isna().sum()
            if failed_parses > 0:
                print(f"⚠️ {failed_parses} rows failed to parse. Trying alternative format...")
                # Try with seconds
                new_data['DateTime'] = pd.to_datetime(new_data['DateTime'], format='%d-%m-%Y %H:%M:%S', errors='coerce')
                failed_parses = new_data['DateTime'].isna().sum()
                if failed_parses > 0:
                    # Try auto-detection
                    new_data['DateTime'] = pd.to_datetime(new_data['DateTime'], errors='coerce')
            
            # Extract date features
            new_data['Year'] = new_data['DateTime'].dt.year
            new_data['Month'] = new_data['DateTime'].dt.month
            new_data['Day'] = new_data['DateTime'].dt.day
            new_data['Hour'] = new_data['DateTime'].dt.hour
            new_data['DayOfWeek'] = new_data['DateTime'].dt.dayofweek
            
            print(f"✅ Successfully parsed dates")
            print(f"   Date range: {new_data['DateTime'].min()} to {new_data['DateTime'].max()}")
            
            # Drop the original DateTime column
            new_data = new_data.drop(columns=['DateTime'])
        else:
            raise ValueError("CSV must contain a 'DateTime' column")
        
        # ===== Prepare features =====
        feature_cols = ['Temperature', 'Humidity', 'Wind_Speed', 
                        'general_diffuse_flows', 'Year', 'Month', 'Day', 
                        'Hour', 'DayOfWeek']
        
        # Check for required columns
        missing_cols = [col for col in feature_cols if col not in new_data.columns]
        if missing_cols:
            raise ValueError(
                f"Missing required columns: {missing_cols}\n"
                f"Available columns: {list(new_data.columns)}\n\n"
                f"Your CSV should have columns: DateTime, Temperature, Humidity, Wind Speed, general diffuse flows, Zone 1"
            )
        
        # Prepare features and target
        X = new_data[feature_cols]
        y = new_data['Zone 1']
        
        # Convert to numeric
        X = X.apply(pd.to_numeric, errors='coerce')
        y = pd.to_numeric(y, errors='coerce')
        
        # Remove NaN
        initial_rows = len(X)
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        
        print(f"\n🧹 Data cleaning:")
        print(f"   Initial rows: {initial_rows}")
        print(f"   Valid rows: {len(X)}")
        print(f"   Removed: {initial_rows - len(X)} rows")
        
        if len(X) < 100:
            raise ValueError(f"Need at least 100 valid samples. Got {len(X)}")
        
        # ===== Train the model =====
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import r2_score, mean_squared_error
        
        print(f"\n🚀 Training Random Forest model...")
        
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
        
        print(f"\n📈 Model Performance:")
        print(f"   R² Score: {r2:.4f}")
        print(f"   RMSE: {rmse:.4f}")
        
        # Save training info
        training_history.append({
            "timestamp": datetime.now().isoformat(),
            "samples": len(X),
            "r2_score": r2,
            "rmse": rmse
        })
        
        # Save model files
        saved_model_dir = os.path.join(current_dir, "saved_model")
        os.makedirs(saved_model_dir, exist_ok=True)
        
        joblib.dump(new_model, os.path.join(saved_model_dir, "Random Forest_model.pkl"))
        joblib.dump(new_scaler, os.path.join(saved_model_dir, "scaler.pkl"))
        joblib.dump(feature_cols, os.path.join(saved_model_dir, "feature_names.pkl"))
        
        print(f"\n💾 Model saved to: {saved_model_dir}")
        
        # Update global variables
        global model, scaler, features, last_training_time
        model = new_model
        scaler = new_scaler
        features = feature_cols
        last_training_time = datetime.now()
        
        print("\n" + "=" * 50)
        print("✅ Model retraining completed successfully!")
        print("=" * 50)
        
        return RetrainResponse(
            status="success",
            message=f"Model retrained with {len(X)} samples. R² Score: {r2:.4f}",
            new_r2_score=r2,
            samples_used=len(X),
            timestamp=datetime.now().isoformat()
        )
        
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Data error: {str(e)}")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Retraining error: {str(e)}")


# ============================================
# Run the API
# ============================================

# Load model on startup
load_model_artifacts()

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)