# 🚗 US Accident Severity Prediction - Streamlit Application

## Project Summary

This package contains a complete Streamlit web application for predicting traffic accident severity using machine learning, integrated with real-time APIs for weather and road feature data.

---

## 📦 Files Included

### Core Application Files

1. **app.py** (Main Application)
   - Complete Streamlit app with 3 pages: Home, Dashboard, Prediction
   - Integrated NOAA Weather API for real-time weather data
   - Integrated OpenStreetMap API for road features
   - Interactive location picker with map
   - Full prediction interface with visualization
   - 500+ lines of production-ready code

2. **utils.py** (Utility Functions)
   - WeatherAPI class for NOAA integration
   - RoadFeaturesAPI class for OpenStreetMap integration
   - FeatureEngineering class for data transformation
   - PredictionVisualizer class for result display
   - Validation utilities
   - 400+ lines of modular, reusable code

3. **dashboard.py** (Sample Dashboard)
   - Example dashboard with visualizations
   - Time series analysis
   - State-wise statistics
   - Weather impact analysis
   - Ready to integrate your actual dashboard

### Configuration Files

4. **config_template.py**
   - Configuration template for API keys
   - Feature definitions
   - Model paths
   - Severity level mappings

5. **requirements.txt**
   - All Python dependencies
   - Tested versions for compatibility

### Documentation

6. **README.md**
   - Comprehensive documentation
   - Installation instructions
   - API integration guide
   - Feature descriptions
   - Troubleshooting section

7. **QUICKSTART.md**
   - 5-minute setup guide
   - Step-by-step instructions
   - Sample use cases
   - Pro tips

8. **setup.py**
   - Automated setup script
   - Dependency checker
   - Configuration helper
   - Directory creation

---

## 🎯 Key Features

### 1. Multi-Page Navigation
- Clean home page with navigation buttons
- Integrated dashboard for analytics
- Dedicated prediction page

### 2. Real-Time Data Integration
- **NOAA Weather API**:
  - Temperature, humidity, pressure
  - Visibility, wind speed/direction
  - Weather conditions
  
- **OpenStreetMap API**:
  - Road infrastructure features
  - Traffic signals, crossings
  - Junctions, amenities
  - Railway crossings

### 3. Intelligent Prediction
- Combines multiple data sources
- Automatic feature engineering
- Temporal features (rush hour, season, etc.)
- Distance-based features
- XGBoost model integration

### 4. User-Friendly Interface
- Interactive map for location selection
- Date/time pickers
- Real-time vs manual input toggle
- Advanced options for power users
- Clear result visualization

### 5. Robust Error Handling
- API failure fallbacks
- Default values when APIs unavailable
- Helpful error messages
- Validation at every step

---

## 🚀 Quick Setup

### Option 1: Automated Setup (Recommended)

```bash
# 1. Run setup script
python setup.py

# 2. Edit config.py with your NOAA API token

# 3. Run the app
streamlit run app.py
```

### Option 2: Manual Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Create config file
cp config_template.py config.py

# 3. Edit config.py with API token

# 4. Ensure model files exist:
#    - model.pkl
#    - preprocessing/new/preprocessor.pkl

# 5. Run the app
streamlit run app.py
```

---

## 📋 Prerequisites

### Required Files (You Must Provide)
- ✅ `model.pkl` - Your trained XGBoost model
- ✅ `preprocessor.pkl` - Your fitted preprocessing pipeline

### Required Setup
- ✅ Python 3.8 or higher
- ✅ NOAA API token (free from https://www.ncdc.noaa.gov/cdo-web/token)
- ✅ Internet connection for API calls

### Included in Package
- ✅ Complete Streamlit application
- ✅ API integration code
- ✅ Utility functions
- ✅ Configuration templates
- ✅ Documentation
- ✅ Setup scripts

---

## 🎨 Application Flow

### Home Page
```
User opens app → Sees welcome screen → Chooses:
  ├─ Dashboard (view analytics)
  └─ Prediction (make new prediction)
```

### Prediction Flow
```
User enters location data (lat, lng, state)
                ↓
User selects date/time
                ↓
App fetches weather data (NOAA API)
                ↓
App fetches road features (OpenStreetMap API)
                ↓
App engineers temporal & distance features
                ↓
Model predicts severity (1-4)
                ↓
User sees results with confidence scores
```

---

## 🔧 Customization Points

### Easy Customizations

1. **Severity Descriptions**: Edit `config_template.py` → `SEVERITY_LEVELS`
2. **Default Values**: Edit functions in `utils.py`
3. **Search Radius**: Edit `OSM_SEARCH_RADIUS` in config
4. **Map Center**: Change default lat/lng in app.py
5. **Color Scheme**: Edit CSS in app.py `st.markdown()` sections

### Advanced Customizations

1. **Dashboard**: Replace `dashboard.py` with your actual dashboard
2. **Additional Features**: Add to `FeatureEngineering` class in utils.py
3. **New APIs**: Add new API classes to utils.py
4. **Validation Rules**: Extend validation functions
5. **Visualization**: Modify `PredictionVisualizer` class

---

## 📊 API Integration Details

### NOAA Weather API

**Endpoint**: `https://www.ncdc.noaa.gov/cdo-web/api/v2/data`

**Authentication**: Token-based (in header)

**Data Retrieved**:
- Temperature (°F)
- Humidity (%)
- Pressure (inches)
- Visibility (miles)
- Wind speed (mph)
- Wind direction
- Weather condition

**Fallback**: Uses default values if API fails

### OpenStreetMap Overpass API

**Endpoint**: `http://overpass-api.de/api/interpreter`

**Authentication**: None required

**Data Retrieved**:
- Amenities within radius
- Road crossings
- Traffic signals
- Junctions
- Stop signs
- Give way signs
- Railway features
- No exit roads

**Radius**: 100 meters (configurable)

**Fallback**: Uses default false values if API fails

---

## 🎯 Model Input Features

The application generates 26 features for prediction:

### Numeric Features (13)
- Start_Lat, Start_Lng
- Temperature(F), Humidity(%), Pressure(in)
- Visibility(mi), Wind_Speed(mph)
- hour, day, month, dayofweek
- Distance(mi)_capped, Distance(mi)_log

### Boolean Features (11)
- Amenity, Crossing, Give_Way, Junction
- No_Exit, Railway, Station, Stop
- Traffic_Signal, is_rushhour, is_holiday

### Categorical Features (6)
- State, Weather_Simple, Wind_Direction_Simple
- time_bucket, Distance(mi)_bin, season

---

## ⚡ Performance Optimization

### Caching
- Model and preprocessor loaded once with `@st.cache_resource`
- Dashboard data cached with `@st.cache_data`
- Reduces load times for subsequent requests

### Efficient API Calls
- Parallel data fetching when possible
- Timeout limits to prevent hanging
- Fallback to defaults on failure

### User Experience
- Loading indicators for long operations
- Progress status for API calls
- Instant prediction with pre-loaded model

---

## 🔒 Security & Privacy

### Data Handling
- No permanent storage of user inputs
- No logging of personal information
- All predictions are ephemeral

### API Security
- API keys stored in config (not in code)
- Rate limiting awareness
- Proper error handling for failed calls

### Best Practices
- Input validation at every step
- Coordinate bounds checking
- Secure API token management

---

## 📈 Future Enhancements

### Suggested Improvements

1. **Enhanced Dashboard**
   - Real-time accident feed
   - Interactive heatmap
   - Predictive analytics

2. **Additional APIs**
   - Traffic API integration
   - Historical weather API
   - Holiday calendar API

3. **Advanced Features**
   - Batch prediction mode
   - Export results to CSV
   - Prediction history
   - Model comparison

4. **User Experience**
   - Dark mode toggle
   - Custom themes
   - Mobile optimization
   - Saved locations

---

## 🐛 Known Limitations

1. **API Rate Limits**
   - NOAA: Limited requests per day
   - OpenStreetMap: Community API with limits
   - Solution: Fallback to defaults

2. **Model Dependencies**
   - Requires exact feature set
   - Version compatibility with scikit-learn/xgboost
   - Solution: Document versions in requirements.txt

3. **Geographic Coverage**
   - NOAA API may have limited historical data
   - OpenStreetMap coverage varies by region
   - Solution: Most comprehensive in US

---

## 📞 Support & Maintenance

### For Issues

1. **Setup Problems**: Run `python setup.py` and check output
2. **API Issues**: Verify tokens and internet connection
3. **Model Issues**: Ensure correct file paths and versions
4. **Import Errors**: Reinstall requirements

### For Questions

- Check README.md for detailed documentation
- Check QUICKSTART.md for setup help
- Review code comments in app.py and utils.py

---

## ✅ Checklist Before Running

- [ ] Python 3.8+ installed
- [ ] All requirements installed (`pip install -r requirements.txt`)
- [ ] config.py created with NOAA API token
- [ ] model.pkl in root directory
- [ ] preprocessor.pkl in preprocessing/new/
- [ ] Internet connection available

---

## 🎓 Learning Resources

### Understanding the Code

- **app.py**: Main application logic and UI
- **utils.py**: Reusable functions and classes
- **dashboard.py**: Visualization examples

### Streamlit Concepts Used

- Session state for navigation
- Caching for performance
- Custom CSS for styling
- Layout columns and containers
- Interactive widgets

### APIs Used

- REST API calls with requests
- JSON data handling
- Error handling and fallbacks
- Rate limiting awareness

---

## 🏆 Success Metrics

### Application Performance
- Load time: < 2 seconds
- Prediction time: < 5 seconds (with API calls)
- API success rate: > 90%

### User Experience
- Intuitive navigation
- Clear error messages
- Helpful fallbacks
- Visual feedback

---

## 📝 Version History

**Version 1.0.0** (Current)
- Complete Streamlit application
- NOAA Weather API integration
- OpenStreetMap API integration
- Interactive prediction interface
- Sample dashboard
- Comprehensive documentation

---

## 🙏 Acknowledgments

- **Streamlit**: Web framework
- **NOAA**: Weather data API
- **OpenStreetMap**: Geographic data
- **XGBoost**: Machine learning model
- **Folium**: Interactive maps

---

## 📄 License

This project is for educational and research purposes.

---

## 🎉 You're All Set!

You now have everything needed to run a professional accident severity prediction system with real-time data integration.

**Next Steps:**
1. Run `python setup.py` to verify installation
2. Configure your NOAA API token
3. Run `streamlit run app.py`
4. Start making predictions!

**Happy Predicting! 🚗🎯**
