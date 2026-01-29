# US Accident Severity Prediction System

A comprehensive Streamlit web application for predicting accident severity using machine learning, integrated with real-time weather data from NOAA API and road features from OpenStreetMap API.

## 🌟 Features

- **Interactive Prediction Interface**: Easy-to-use form for inputting accident details
- **Real-time Data Integration**: 
  - NOAA API for weather conditions
  - OpenStreetMap API for road features
- **Interactive Map**: Visualize accident location
- **Dashboard**: View historical accident data and analytics
- **Comprehensive Analysis**: Multiple factors including location, weather, time, and road conditions

## 📋 Prerequisites

- Python 3.8 or higher
- Trained XGBoost model (`model.pkl`)
- Preprocessor object (`preprocessor.pkl`)
- NOAA API token (free from https://www.ncdc.noaa.gov/cdo-web/token)

## 🚀 Installation

1. **Clone or download the project files**

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Set up API keys**:
   - Copy `config_template.py` to `config.py`
   - Get your NOAA API token from https://www.ncdc.noaa.gov/cdo-web/token
   - Update `config.py` with your NOAA API token

4. **Ensure model files are in place**:
   - `model.pkl` should be in the root directory
   - `preprocessor.pkl` should be in `preprocessing/new/` directory

## 🎮 Usage

1. **Run the Streamlit app**:
```bash
streamlit run app.py
```

2. **Navigate the application**:
   - **Home Page**: Choose between Dashboard or Prediction
   - **Dashboard**: View accident statistics and visualizations
   - **Prediction Page**: Enter accident details and get severity prediction

## 📊 Prediction Page

### Input Fields:

**Location Information:**
- State (dropdown selection)
- Latitude and Longitude
- Distance affected by accident
- Interactive map to visualize location

**Time Information:**
- Accident date
- Accident time

**Data Source Options:**
- Fetch real-time data from APIs (recommended)
- Manual input override (advanced users)

### Output:

- **Predicted Severity Level**: 1-4 (Low to Severe)
- **Confidence Scores**: Probability for each severity level
- **Visual Indicators**: Color-coded severity display
- **Input Summary**: Review of all input parameters

## 🔧 API Integration Details

### NOAA Weather API

Fetches real-time weather conditions:
- Temperature (°F)
- Humidity (%)
- Atmospheric Pressure (in)
- Visibility (mi)
- Wind Speed (mph)
- Weather Condition
- Wind Direction

**Note**: You need a free API token from NOAA. The API has rate limits, so the app includes fallback to default values if the API is unavailable.

### OpenStreetMap Overpass API

Fetches road infrastructure features within 100m radius:
- Amenities nearby
- Road crossings
- Give way signs
- Junctions
- Stop signs
- Traffic signals
- Railway crossings
- Railway stations

**Note**: No API key required. Public API with usage limits.

## 🗂️ Project Structure

```
.
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── config_template.py          # Configuration template
├── config.py                   # Your actual config (create from template)
├── model.pkl                   # Trained XGBoost model
├── preprocessing/
│   └── new/
│       └── preprocessor.pkl    # Fitted preprocessing pipeline
└── README.md                   # This file
```

## 🧮 Model Features

The model uses 26 features across 3 categories:

### Numeric Features (13):
- Geographic coordinates (Lat, Lng)
- Weather metrics (Temperature, Humidity, Pressure, Visibility, Wind Speed)
- Temporal features (hour, day, month, dayofweek)
- Distance metrics (capped, log-transformed)

### Boolean Features (11):
- Road infrastructure flags
- Traffic control indicators
- Rush hour and holiday flags

### Categorical Features (6):
- State
- Weather condition
- Wind direction
- Time bucket
- Distance bin
- Season

## 🎯 Severity Levels

1. **Low (🟢)**: Minor accident with minimal traffic impact
2. **Moderate (🟡)**: Moderate accident with some traffic disruption
3. **High (🟠)**: Serious accident causing significant delays
4. **Severe (🔴)**: Critical accident with major traffic impact

## ⚙️ Configuration

Edit `config.py` to customize:
- API endpoints and timeouts
- Model and preprocessor paths
- Feature lists
- Severity level descriptions
- Search radius for OpenStreetMap queries

## 🐛 Troubleshooting

### API Errors:
- **NOAA API fails**: Check your API token, ensure it's valid
- **OpenStreetMap fails**: Check internet connection, API might be rate-limited
- **Default values used**: App will use reasonable defaults if APIs fail

### Model Errors:
- Ensure `model.pkl` and `preprocessor.pkl` exist in correct locations
- Check that files were created with compatible scikit-learn/xgboost versions

### Import Errors:
- Run `pip install -r requirements.txt` to install all dependencies
- Check Python version (3.8+ required)

## 📝 Notes

- The app includes manual input override for testing without API access
- Map visualization requires internet connection
- First load may be slower as Streamlit caches the model
- API rate limits may affect real-time data fetching frequency

## 🔐 Privacy & Data

- No user data is stored permanently
- All predictions are made locally
- API calls only fetch publicly available data
- Location data is not shared with third parties

## 📧 Support

For issues or questions:
1. Check the troubleshooting section
2. Verify API keys and model files
3. Check Streamlit documentation: https://docs.streamlit.io

## 📄 License

This project is for educational and research purposes.

## 🙏 Acknowledgments

- NOAA for weather data API
- OpenStreetMap for road feature data
- Streamlit for the web framework
- US Accidents dataset creators
