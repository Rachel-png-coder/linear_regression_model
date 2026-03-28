"""
Power Consumption Prediction API
FastAPI endpoint for predicting building energy usage
Supports multiple date formats including AM/PM
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
    description="Predict building power consumption using environmental factors. Supports AM/PM date formats.",
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
    Parse dates with multiple possible formats including AM/PM
    Handles various date formats and time representations
    """
    
    print(f"🔍 Original date sample: {df[date_column].iloc[0] if len(df) > 0 else 'No data'}")
    
    # Step 1: Clean the data - remove extra spaces and standardize
    df[date_column] = df[date_column].astype(str).str.strip()
    
    # Step 2: Define comprehensive format list
    date_formats = []
    
    # Date part formats
    date_patterns = [
        ('%m', '%d', '%Y'),  # MM/DD/YYYY
        ('%d', '%m', '%Y'),  # DD/MM/YYYY
        ('%Y', '%m', '%d'),  # YYYY/MM/DD
    ]
    
    # Separators
    separators = ['/', '-']
    
    # Time formats to try (including AM/PM)
    time_formats = [
        '',                    # No time
        ' %H:%M',             # 24-hour without seconds
        ' %H:%M:%S',          # 24-hour with seconds
        ' %I:%M %p',          # 12-hour with AM/PM (space)
        ' %I:%M:%S %p',       # 12-hour with seconds and AM/PM
        ' %I:%M%p',           # 12-hour without space (12:00AM)
        ' %I%p',              # 12-hour hour only (12AM)
        ' %I %p',             # 12-hour with space (12 AM)
        ' %I:%M%p',           # 12-hour with AM/PM no space
        ' %I:%M:%S%p',        # 12-hour with seconds no space
    ]
    
    # Build all format combinations
    for date_parts in date_patterns:
        for sep in separators:
            # Date only formats
            date_format = sep.join(date_parts)
            date_formats.append(date_format)
            
            # Date + time formats
            for time_format in time_formats:
                full_format = date_format + time_format
                date_formats.append(full_format)
    
    # Add common formats with spaces
    common_formats = [
        '%m/%d/%Y %I:%M %p',
        '%m/%d/%Y %I:%M:%S %p',
        '%m/%d/%Y %I:%M%p',
        '%m/%d/%Y %I%p',
        '%d/%m/%Y %I:%M %p',
        '%d/%m/%Y %I:%M:%S %p',
        '%Y-%m-%d %I:%M %p',
        '%Y-%m-%d %I:%M:%S %p',
        '%m-%d-%Y %I:%M %p',
        '%d-%m-%Y %I:%M %p',
        '%m/%d/%Y %H:%M',
        '%m/%d/%Y %H:%M:%S',
        '%d/%m/%Y %H:%M',
        '%d/%m/%Y %H:%M:%S',
        '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%d %H:%M',
        '%m/%d/%Y',
        '%d/%m/%Y',
        '%Y-%m-%d',
    ]
    
    date_formats.extend(common_formats)
    
    # Remove duplicates while preserving order
    seen = set()
    date_formats = [x for x in date_formats if not (x in seen or seen.add(x))]
    
    # Step 3: Try each format
    print(f"🔄 Attempting to parse dates (trying {len(date_formats)} formats)...")
    
    for date_format in date_formats:
        try:
            df[date_column] = pd.to_datetime(df[date_column], format=date_format)
            print(f"✅ Successfully parsed with format: {date_format}")
            return df
        except (ValueError, TypeError):
            continue
    
    # Step 4: Try pandas flexible parsing with mixed formats
    try:
        print("🔄 Trying pandas flexible parsing with dayfirst=False...")
        df[date_column] = pd.to_datetime(df[date_column], format='mixed', dayfirst=False)
        print("✅ Successfully parsed with mixed format (MM/DD/YYYY)")
        return df
    except Exception:
        pass
    
    try:
        print("🔄 Trying pandas flexible parsing with dayfirst=True...")
        df[date_column] = pd.to_datetime(df[date_column], format='mixed', dayfirst=True)
        print("✅ Successfully parsed with mixed format (DD/MM/YYYY)")
        return df
    except Exception:
        pass
    
    # Step 5: Try manual conversion using dateutil for stubborn formats
    try:
        print("🔄 Attempting manual date parsing with dateutil...")
        from dateutil import parser
        
        def manual_parse(date_str):
            try:
                # Try parsing with fuzzy matching for AM/PM
                return parser.parse(date_str, fuzzy=True)
            except:
                return pd.NaT
        
        df[date_column] = df[date_column].apply(manual_parse)
        if df[date_column].isna().sum() < len(df) * 0.2:  # Less than 20% failed
            print("✅ Successfully parsed using manual dateutil parser")
            return df
    except Exception as e:
        print(f"⚠️ Manual parsing failed: {e}")
    
    # Step 6: If all attempts fail, provide detailed error
    sample_values = df[date_column].head(5).tolist()
    error_msg = (
        f"❌ Unable to parse dates in column '{date_column}'.\n"
        f"Sample values (first 5):\n"
    )
    for i, val in enumerate(sample_values, 1):
        error_msg += f"  {i}. '{val}'\n"
    
    error_msg += (
        f"\n✅ Supported formats include:\n"
        f"  • MM/DD/YYYY HH:MM AM/PM (e.g., 1/13/2017 12:00 AM)\n"
        f"  • DD/MM/YYYY HH:MM AM/PM (e.g., 13/1/2017 12:00 PM)\n"
        f"  • YYYY-MM-DD HH:MM:SS (e.g., 2017-01-13 00:00:00)\n"
        f"  • MM/DD/YYYY HH:MM (e.g., 1/13/2017 14:30)\n"
        f"  • And many other variations\n"
    )
    
    raise ValueError(error_msg)


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
        print(f"Current working directory: {os.getcwd()}")
        
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
            print(f"Files in current directory: {os.listdir('.')}")
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
    """Retrain model with new data with enhanced AM/PM support"""
    global model, scaler, features, last_training_time, training_history
    
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_squared_error
    from sklearn.preprocessing import StandardScaler
    
    print("\n" + "=" * 50)
    print("Starting Model Retraining...")
    print("=" * 50)
    
    # ===== ENHANCED: Find date column with multiple variations =====
    print("\n📅 Processing dates...")
    
    date_column_candidates = ['DateTime', 'Date', 'Time', 'Datetime', 'date', 'datetime', 'timestamp', 'Timestamp']
    date_column = None
    
    for candidate in date_column_candidates:
        if candidate in new_data.columns:
            date_column = candidate
            break
    
    if date_column:
        print(f"📅 Found date column: '{date_column}'")
        print(f"📅 Sample values: {new_data[date_column].head(3).tolist()}")
        
        # Parse dates with AM/PM support
        new_data = parse_dates_flexible(new_data, date_column)
        
        # Extract features from DateTime
        new_data['Year'] = new_data[date_column].dt.year
        new_data['Month'] = new_data[date_column].dt.month
        new_data['Day'] = new_data[date_column].dt.day
        new_data['Hour'] = new_data[date_column].dt.hour
        new_data['DayOfWeek'] = new_data[date_column].dt.dayofweek
        
        print(f"✅ Extracted date features: Year, Month, Day, Hour, DayOfWeek")
        print(f"📊 Date range: {new_data[date_column].min()} to {new_data[date_column].max()}")
    else:
        print("⚠️ No date column found")
        # Check if date components already exist
        if 'Year' not in new_data.columns:
            raise ValueError(
                "No date column found and 'Year' column missing. "
                "Please provide DateTime column or date components (Year, Month, Day, Hour)."
            )
        else:
            print("✅ Using existing date components")
    
    # Handle column name variations
    column_mappings = {
        'Wind Speed': 'Wind_Speed',
        'Wind_Speed': 'Wind_Speed',
        'general diffuse flows': 'general_diffuse_flows',
        'general_diffuse_flows': 'general_diffuse_flows',
        'Temperature': 'Temperature',
        'Humidity': 'Humidity',
        'Zone 1': 'Zone 1',
        'Zone1': 'Zone 1'
    }
    
    for old_name, new_name in column_mappings.items():
        if old_name in new_data.columns and old_name != new_name:
            new_data = new_data.rename(columns={old_name: new_name})
            print(f"📝 Renamed column: '{old_name}' -> '{new_name}'")
    
    # Define features
    feature_cols = ['Temperature', 'Humidity', 'Wind_Speed', 
                    'general_diffuse_flows', 'Year', 'Month', 'Day', 
                    'Hour', 'DayOfWeek']
    
    print(f"\n📊 Required features: {feature_cols}")
    print(f"📊 Available columns: {list(new_data.columns)}")
    
    # Check if all required columns exist
    missing_cols = [col for col in feature_cols if col not in new_data.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns: {missing_cols}\n"
            f"Available columns: {list(new_data.columns)}"
        )
    
    # Prepare features and target
    X = new_data[feature_cols]
    y = new_data['Zone 1']
    
    # Remove NaN
    initial_rows = len(X)
    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask]
    y = y[mask]
    
    print(f"\n🧹 Data cleaning:")
    print(f"   Initial rows: {initial_rows}")
    print(f"   After removing NaN: {len(X)} rows")
    print(f"   Removed: {initial_rows - len(X)} rows")
    
    if len(X) < 100:
        raise ValueError(f"Need at least 100 samples for retraining. Got {len(X)} samples.")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"\n📊 Data split:")
    print(f"   Training samples: {len(X_train)}")
    print(f"   Testing samples: {len(X_test)}")
    
    # Scale
    new_scaler = StandardScaler()
    X_train_scaled = new_scaler.fit_transform(X_train)
    X_test_scaled = new_scaler.transform(X_test)
    
    # Train
    print(f"\n🚀 Training Random Forest model...")
    new_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1,
        verbose=0
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
    
    # Save model files with absolute path
    saved_model_dir = os.path.join(current_dir, "saved_model")
    os.makedirs(saved_model_dir, exist_ok=True)
    
    joblib.dump(new_model, os.path.join(saved_model_dir, "Random Forest_model.pkl"))
    joblib.dump(new_scaler, os.path.join(saved_model_dir, "scaler.pkl"))
    joblib.dump(feature_cols, os.path.join(saved_model_dir, "feature_names.pkl"))
    
    print(f"\n💾 Model saved to: {saved_model_dir}")
    
    # Update globals
    model = new_model
    scaler = new_scaler
    features = feature_cols
    last_training_time = datetime.now()
    
    print("\n" + "=" * 50)
    print("✅ Model retraining completed successfully!")
    print("=" * 50)
    
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
        "date_format_support": "Full AM/PM support for all common date formats",
        "endpoints": {
            "predict": "/predict (POST)",
            "retrain": "/retrain (POST) - Supports AM/PM dates",
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
    Supports dates with AM/PM formats like:
    - 1/13/2017 12:00 AM
    - 13/1/2017 2:30 PM
    - 2017-01-13 11:59 PM
    - And many more formats
    """
    
    try:
        # Read uploaded file
        contents = await file.read()
        new_data = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        print(f"\n📊 Received file: {file.filename}")
        print(f"📊 Total rows: {len(new_data)}")
        print(f"📊 Columns: {list(new_data.columns)}")
        
        # Validate required columns
        required_cols = ['Zone 1', 'Temperature', 'Humidity']
        missing_cols = [col for col in required_cols if col not in new_data.columns]
        
        if missing_cols:
            # Try case-insensitive matching
            for col in missing_cols:
                found = False
                for existing_col in new_data.columns:
                    if existing_col.lower() == col.lower():
                        # Rename the column
                        new_data = new_data.rename(columns={existing_col: col})
                        found = True
                        print(f"📝 Renamed column: '{existing_col}' -> '{col}'")
                if not found:
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Missing required column: {col}. Available columns: {list(new_data.columns)}"
                    )
        
        # Clean data
        initial_rows = len(new_data)
        new_data = new_data.dropna()
        cleaned_rows = len(new_data)
        
        print(f"\n🧹 Data cleaning:")
        print(f"   Initial rows: {initial_rows}")
        print(f"   After removing NaN: {cleaned_rows} rows")
        print(f"   Removed: {initial_rows - cleaned_rows} rows")
        
        if cleaned_rows < 100:
            raise HTTPException(
                status_code=400,
                detail=f"Need at least 100 samples for retraining. Got {cleaned_rows} samples."
            )
        
        # Retrain with AM/PM support
        _, _, r2, samples_used = retrain_model_with_new_data(new_data)
        
        return RetrainResponse(
            status="success",
            message=f"Model retrained with {samples_used} samples. R² Score: {r2:.4f}",
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