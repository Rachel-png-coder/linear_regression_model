# Power Consumption Prediction
This project predicts building power consumption using environmental factors (temperature, 
humidity, wind speed, solar radiation). Three regression models are compared with Random Forest
 performing best (R² ≈ 0.89). The model enables real-time energy forecasting for grid management.

Data Source: https://storage.googleapis.com/kagglesdsdata/datasets/6484787/10473366/power%20consumption.csv?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20260313%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20260313T082405Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=95f3d2f93aef89500f6f6949d6938662118fafde53a11d3ed76c95d9d01c95f6a6ef5aa2c62aa9f633a166cf4f3b59a6a2c644ca24c85ca005a665eb6e81b1034c5a5a9f99771b7cd0ed1fca1b188418af8fb6169c7efa7816f92519e4dd62224e4b4c276a8eef733db46e62b6148472164a1483652e22d8042be91f2e2a87b264920ffabedd06e26c8601b8b7e6e8d43ba40884757e66bf7660112d43a87bf3af7e357a7e94d10aa68b13082c6263ed647e7c0bc7978bef2cf215a37b08dd9340fa13e9541762b97bfe3e5b4737b255307e4753f6061dfbdf894ad6847ce3c3226c389cfe9b6b3fd45ae76a8cf335cb3672fd9827b2f65b3db1243cbdfddf6c

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
