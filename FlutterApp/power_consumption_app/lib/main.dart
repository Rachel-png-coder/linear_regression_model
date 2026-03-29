import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

void main() {
  runApp(const PowerConsumptionApp());
}

class PowerConsumptionApp extends StatelessWidget {
  const PowerConsumptionApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Power Consumption Predictor',
      theme: ThemeData(primarySwatch: Colors.blue, useMaterial3: true),
      debugShowCheckedModeBanner: false,
      home: const PredictionPage(),
    );
  }
}

class PredictionPage extends StatefulWidget {
  const PredictionPage({super.key});

  @override
  State<PredictionPage> createState() => _PredictionPageState();
}

class _PredictionPageState extends State<PredictionPage> {
  final String apiUrl = 'https://linear-regression-model-asgf.onrender.com';
  final String apiPath = '/predict';

  final TextEditingController _temperatureController = TextEditingController();
  final TextEditingController _humidityController = TextEditingController();
  final TextEditingController _windSpeedController = TextEditingController();
  final TextEditingController _generalDiffuseFlowsController =
      TextEditingController();
  final TextEditingController _yearController = TextEditingController();
  final TextEditingController _monthController = TextEditingController();
  final TextEditingController _dayController = TextEditingController();
  final TextEditingController _hourController = TextEditingController();
  final TextEditingController _dayOfWeekController = TextEditingController();

  String _predictionResult = '';
  bool _isLoading = false;
  String _errorMessage = '';

  @override
  void dispose() {
    _temperatureController.dispose();
    _humidityController.dispose();
    _windSpeedController.dispose();
    _generalDiffuseFlowsController.dispose();
    _yearController.dispose();
    _monthController.dispose();
    _dayController.dispose();
    _hourController.dispose();
    _dayOfWeekController.dispose();
    super.dispose();
  }

  Future<void> _makePrediction() async {
    setState(() {
      _errorMessage = '';
      _predictionResult = '';
    });

    try {
      double temperature = double.parse(_temperatureController.text.trim());
      double humidity = double.parse(_humidityController.text.trim());
      double windSpeed = double.parse(_windSpeedController.text.trim());
      double generalDiffuseFlows = double.parse(
        _generalDiffuseFlowsController.text.trim(),
      );
      int year = int.parse(_yearController.text.trim());
      int month = int.parse(_monthController.text.trim());
      int day = int.parse(_dayController.text.trim());
      int hour = int.parse(_hourController.text.trim());
      int dayOfWeek = int.parse(_dayOfWeekController.text.trim());

      // Range validation
      if (temperature < -50 || temperature > 60) {
        setState(
          () => _errorMessage = 'Temperature must be between -50°C and 60°C',
        );
        return;
      }
      if (humidity < 0 || humidity > 100) {
        setState(() => _errorMessage = 'Humidity must be between 0% and 100%');
        return;
      }
      if (windSpeed < 0 || windSpeed > 200) {
        setState(
          () => _errorMessage = 'Wind Speed must be between 0 and 200 km/h',
        );
        return;
      }
      if (generalDiffuseFlows < 0 || generalDiffuseFlows > 2000) {
        setState(
          () => _errorMessage =
              'General Diffuse Flows must be between 0 and 2000 W/m²',
        );
        return;
      }
      if (year < 2000 || year > 2030) {
        setState(() => _errorMessage = 'Year must be between 2000 and 2030');
        return;
      }
      if (month < 1 || month > 12) {
        setState(() => _errorMessage = 'Month must be between 1 and 12');
        return;
      }
      if (day < 1 || day > 31) {
        setState(() => _errorMessage = 'Day must be between 1 and 31');
        return;
      }
      if (hour < 0 || hour > 23) {
        setState(() => _errorMessage = 'Hour must be between 0 and 23');
        return;
      }
      if (dayOfWeek < 0 || dayOfWeek > 6) {
        setState(
          () => _errorMessage =
              'Day of week must be between 0 (Monday) and 6 (Sunday)',
        );
        return;
      }

      // Prepare request body
      Map<String, dynamic> requestBody = {
        "temperature": temperature,
        "humidity": humidity,
        "wind_speed": windSpeed,
        "general_diffuse_flows": generalDiffuseFlows,
        "year": year,
        "month": month,
        "day": day,
        "hour": hour,
        "day_of_week": dayOfWeek,
      };

      setState(() => _isLoading = true);

      final response = await http
          .post(
            Uri.parse('$apiUrl$apiPath'),
            headers: {'Content-Type': 'application/json'},
            body: json.encode(requestBody),
          )
          .timeout(const Duration(seconds: 30));

      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        setState(() {
          _predictionResult =
              data['prediction']?.toString() ?? 'No prediction received';
        });
      } else if (response.statusCode == 422) {
        final error = json.decode(response.body);
        setState(() {
          _errorMessage = 'Validation error: ${error['detail']}';
        });
      } else {
        setState(() {
          _errorMessage =
              'Server error (${response.statusCode}). Please try again.';
        });
      }
    } catch (e) {
      setState(() {
        _errorMessage = 'Please enter valid numbers in all fields';
      });
    } finally {
      setState(() => _isLoading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Power Consumption Predictor'),
        centerTitle: true,
        backgroundColor: Colors.blue.shade700,
        elevation: 2,
      ),
      body: Container(
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: [Colors.blue.shade50, Colors.white],
          ),
        ),
        child: SafeArea(
          child: SingleChildScrollView(
            padding: const EdgeInsets.all(20.0),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                // Header Card
                Card(
                  elevation: 4,
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(15),
                  ),
                  child: Padding(
                    padding: const EdgeInsets.all(20.0),
                    child: Column(
                      children: [
                        Icon(
                          Icons.electric_bolt,
                          size: 60,
                          color: Colors.blue.shade700,
                        ),
                        const SizedBox(height: 10),
                        const Text(
                          'Power Consumption Prediction',
                          style: TextStyle(
                            fontSize: 22,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                        const SizedBox(height: 5),
                        Text(
                          'Enter all 9 parameters for accurate prediction',
                          style: TextStyle(
                            fontSize: 14,
                            color: Colors.grey.shade600,
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
                const SizedBox(height: 20),

                // Input Fields Card
                Card(
                  elevation: 3,
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(15),
                  ),
                  child: Padding(
                    padding: const EdgeInsets.all(20.0),
                    child: Column(
                      children: [
                        // Row 1: Temperature, Humidity
                        Row(
                          children: [
                            Expanded(
                              child: _buildTextField(
                                _temperatureController,
                                'Temperature',
                                '°C',
                                Icons.thermostat,
                              ),
                            ),
                            const SizedBox(width: 12),
                            Expanded(
                              child: _buildTextField(
                                _humidityController,
                                'Humidity',
                                '%',
                                Icons.water_drop,
                              ),
                            ),
                          ],
                        ),
                        const SizedBox(height: 16),

                        // Row 2: Wind Speed, General Diffuse Flows
                        Row(
                          children: [
                            Expanded(
                              child: _buildTextField(
                                _windSpeedController,
                                'Wind Speed',
                                'km/h',
                                Icons.air,
                              ),
                            ),
                            const SizedBox(width: 12),
                            Expanded(
                              child: _buildTextField(
                                _generalDiffuseFlowsController,
                                'Gen Diffuse Flows',
                                'W/m²',
                                Icons.show_chart,
                              ),
                            ),
                          ],
                        ),
                        const SizedBox(height: 16),

                        // Row 3: Year, Month, Day
                        Row(
                          children: [
                            Expanded(
                              child: _buildTextField(
                                _yearController,
                                'Year',
                                '',
                                Icons.calendar_today,
                              ),
                            ),
                            const SizedBox(width: 12),
                            Expanded(
                              child: _buildTextField(
                                _monthController,
                                'Month',
                                '',
                                Icons.date_range,
                              ),
                            ),
                            const SizedBox(width: 12),
                            Expanded(
                              child: _buildTextField(
                                _dayController,
                                'Day',
                                '',
                                Icons.today,
                              ),
                            ),
                          ],
                        ),
                        const SizedBox(height: 16),

                        // Row 4: Hour, Day of Week
                        Row(
                          children: [
                            Expanded(
                              child: _buildTextField(
                                _hourController,
                                'Hour',
                                '',
                                Icons.access_time,
                              ),
                            ),
                            const SizedBox(width: 12),
                            Expanded(
                              child: _buildTextField(
                                _dayOfWeekController,
                                'Day of Week (0-6)',
                                '',
                                Icons.weekend,
                              ),
                            ),
                          ],
                        ),
                      ],
                    ),
                  ),
                ),
                const SizedBox(height: 20),

                // Predict Button
                ElevatedButton(
                  onPressed: _isLoading ? null : _makePrediction,
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.blue.shade700,
                    foregroundColor: Colors.white,
                    padding: const EdgeInsets.symmetric(vertical: 16),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12),
                    ),
                    elevation: 3,
                  ),
                  child: _isLoading
                      ? const SizedBox(
                          height: 20,
                          width: 20,
                          child: CircularProgressIndicator(
                            strokeWidth: 2,
                            valueColor: AlwaysStoppedAnimation<Color>(
                              Colors.white,
                            ),
                          ),
                        )
                      : const Text(
                          'Predict Power Consumption',
                          style: TextStyle(
                            fontSize: 18,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                ),
                const SizedBox(height: 20),

                // Results Area
                if (_predictionResult.isNotEmpty || _errorMessage.isNotEmpty)
                  Card(
                    elevation: 3,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(15),
                    ),
                    color: _errorMessage.isNotEmpty
                        ? Colors.red.shade50
                        : Colors.green.shade50,
                    child: Padding(
                      padding: const EdgeInsets.all(20.0),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Row(
                            children: [
                              Icon(
                                _errorMessage.isNotEmpty
                                    ? Icons.error_outline
                                    : Icons.check_circle_outline,
                                color: _errorMessage.isNotEmpty
                                    ? Colors.red.shade700
                                    : Colors.green.shade700,
                                size: 28,
                              ),
                              const SizedBox(width: 12),
                              Text(
                                _errorMessage.isNotEmpty
                                    ? 'Error'
                                    : 'Predicted Power Consumption',
                                style: TextStyle(
                                  fontSize: 18,
                                  fontWeight: FontWeight.bold,
                                  color: _errorMessage.isNotEmpty
                                      ? Colors.red.shade800
                                      : Colors.green.shade800,
                                ),
                              ),
                            ],
                          ),
                          const Divider(),
                          const SizedBox(height: 12),
                          Text(
                            _errorMessage.isNotEmpty
                                ? _errorMessage
                                : _predictionResult,
                            style: TextStyle(
                              fontSize: _errorMessage.isNotEmpty ? 14 : 32,
                              fontWeight: FontWeight.bold,
                              color: _errorMessage.isNotEmpty
                                  ? Colors.red.shade900
                                  : Colors.green.shade900,
                            ),
                          ),
                          if (!_errorMessage.isNotEmpty &&
                              _predictionResult.isNotEmpty)
                            Padding(
                              padding: const EdgeInsets.only(top: 8.0),
                              child: Text(
                                'kilowatt-hours (kWh)',
                                style: TextStyle(
                                  fontSize: 14,
                                  color: Colors.grey.shade600,
                                ),
                              ),
                            ),
                        ],
                      ),
                    ),
                  ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildTextField(
    TextEditingController controller,
    String label,
    String suffix,
    IconData icon,
  ) {
    return TextField(
      controller: controller,
      decoration: InputDecoration(
        labelText: label,
        hintText: 'Enter $label',
        prefixIcon: Icon(icon, color: Colors.blue.shade600),
        suffixText: suffix.isEmpty ? null : suffix,
        border: OutlineInputBorder(borderRadius: BorderRadius.circular(12)),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(12),
          borderSide: BorderSide(color: Colors.blue.shade700, width: 2),
        ),
        filled: true,
        fillColor: Colors.white,
      ),
      keyboardType: TextInputType.number,
    );
  }
}
