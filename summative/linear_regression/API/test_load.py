# test_load.py - Run this first to verify files load correctly
import joblib
import os

# Set the correct path
model_path = "saved_model/Random Forest_model.pkl"
scaler_path = "saved_model/scaler.pkl"
features_path = "saved_model/feature_names.pkl"

print("=" * 50)
print("Testing Model Files")
print("=" * 50)

# Check if files exist
print("\n1. Checking if files exist...")
files_exist = True
for path in [model_path, scaler_path, features_path]:
    if os.path.exists(path):
        size = os.path.getsize(path) / (1024 * 1024)
        print(f"   ✅ {path} ({size:.2f} MB)")
    else:
        print(f"   ❌ {path} NOT FOUND")
        files_exist = False

if not files_exist:
    print("\n❌ Files missing! Make sure the 'saved_model' folder is in the same directory as this script.")
    exit()

# Load files
print("\n2. Loading files...")
try:
    model = joblib.load(model_path)
    print(f"   ✅ Model loaded: {type(model)}")
    
    scaler = joblib.load(scaler_path)
    print(f"   ✅ Scaler loaded: {type(scaler)}")
    
    features = joblib.load(features_path)
    print(f"   ✅ Features loaded: {len(features)} features")
    print(f"      Features: {features}")
    
    print("\n✅ All files loaded successfully!")
    
except Exception as e:
    print(f"\n❌ Error loading files: {e}")