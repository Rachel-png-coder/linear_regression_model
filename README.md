# Power Consumption Prediction
This project predicts building power consumption using environmental factors (temperature, 
humidity, wind speed, solar radiation). Three regression models are compared with Random Forest
 performing best (R² ≈ 0.89). The model enables real-time energy forecasting for grid management.

## Dataset Description & Source

**Dataset**: Appliances Energy Prediction Dataset  
**Source**: UCI Machine Learning Repository  
**Permanent Link**: https://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction  
**Kaggle Mirror**: https://www.kaggle.com/datasets/uciml/appliances-energy-prediction

**Data Collection Period**: 4.5 months (January-May 2017)  
**Recording Frequency**: Every 10 minutes  
**Total Samples**: 19,735 measurements  
**Target Variable**: Power consumption (kWh)

Link to SWAGGER UI: https://linear-regression-model-asgf.onrender.com/docs
link to YouTube Demo: https://youtu.be/xmubHdDNRLE

1. Install Required Software
Before running the app, ensure you have the following installed:

bash
# Check Flutter installation
flutter --version

# If not installed, download from:
# https://docs.flutter.dev/get-started/install
2. Verify Flutter Setup
bash
# Run Flutter doctor to ensure everything is configured
flutter doctor

# You should see all checks passed with green checkmarks
Step-by-Step Instructions
Step 1: Clone or Navigate to Your Project
bash
# Navigate to your Flutter project directory
cd C:\Users\LENOVO\Desktop\power_consumption_app
Step 2: Install Dependencies
bash
# Get all required packages
flutter pub get
This installs the HTTP package needed for API calls.

Step 3: Verify API Endpoint
Open lib/main.dart and confirm the API URL is correct:

dart
final String apiUrl = 'https://linear-regression-model-asgf.onrender.com';
final String apiPath = '/predict';
Step 4: Run the App
Option A: Run on Android Emulator
Start an Android emulator:

bash
# List available emulators
flutter emulators

# Launch an emulator (replace <emulator-id> with actual ID)
flutter emulators --launch <emulator-id>
Run the app:

bash
flutter run

# Run the app
flutter run
Option C: Run on Chrome (Web - Easiest for Testing)
bash
# Enable web support (if not already enabled)
flutter config --enable-web

# Run on Chrome
flutter run -d chrome
Option D: Run on Physical Device
Android:

Enable Developer Options and USB Debugging on your Android phone

Connect via USB cable

Run:

bash
flutter run
