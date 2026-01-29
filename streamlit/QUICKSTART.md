# Quick Start Guide

## 🚀 Get Started in 5 Minutes

### Step 1: Install Dependencies (1 minute)

```bash
pip install -r requirements.txt
```

Or run the setup script:

```bash
python setup.py
```

### Step 2: Configure API Access (2 minutes)

1. Copy the configuration template:
   ```bash
   cp config_template.py config.py
   ```

2. Get your free NOAA API token:
   - Visit: https://www.ncdc.noaa.gov/cdo-web/token
   - Enter your email
   - Receive token via email (instant)

3. Edit `config.py` and add your token:
   ```python
   NOAA_API_TOKEN = "your_token_here"
   ```

### Step 3: Verify Model Files (30 seconds)

Ensure these files exist:
- ✅ `model.pkl` (in root directory)
- ✅ `preprocessing/new/preprocessor.pkl`

### Step 4: Run the App (30 seconds)

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Step 5: Make Your First Prediction (1 minute)

1. Click **"🔮 Make Prediction"**
2. Enter location details (or use defaults)
3. Select date and time
4. Click **"🎯 Predict Severity"**
5. View results!

---

## 📱 Application Features

### Home Page
- Quick access to Dashboard and Prediction
- Application overview

### Dashboard Page
- View accident statistics
- Analyze trends and patterns
- Explore visualizations

### Prediction Page
- Enter accident details
- Auto-fetch weather data (NOAA API)
- Auto-fetch road features (OpenStreetMap)
- Get instant severity prediction
- View confidence scores

---

## 🔧 Troubleshooting

### "Module not found" errors
```bash
pip install -r requirements.txt
```

### "Model file not found"
Ensure `model.pkl` and `preprocessor.pkl` are in correct locations

### NOAA API not working
- Check your API token in `config.py`
- Verify internet connection
- App will use default weather values if API fails

### OpenStreetMap API not working
- Check internet connection
- API has rate limits (wait a moment between requests)
- App will use default road features if API fails

---

## 💡 Tips for Best Results

### Location Input
- Use accurate latitude/longitude for best predictions
- Click on the map to adjust location
- Select the correct state

### Time Input
- Consider rush hour times (7-9 AM, 4-7 PM)
- Weekend vs weekday patterns differ
- Time of year affects weather patterns

### API Data
- Enable "Fetch real-time data" for accurate predictions
- APIs may have delays or rate limits
- Manual override available for testing

### Understanding Results
- **Severity 1 (🟢 Low)**: Minor impact, short delays
- **Severity 2 (🟡 Moderate)**: Some disruption expected
- **Severity 3 (🟠 High)**: Significant traffic impact
- **Severity 4 (🔴 Severe)**: Critical, major disruptions

---

## 📊 Sample Use Cases

### Use Case 1: Rush Hour Prediction
```
Location: Los Angeles, CA
Coordinates: 34.0522°N, 118.2437°W
Time: 5:30 PM on a weekday
Weather: Clear, 72°F
Result: Predict severity for typical evening commute
```

### Use Case 2: Weather Impact Analysis
```
Location: Boston, MA
Coordinates: 42.3601°N, 71.0589°W
Time: 8:00 AM, snowy conditions
Weather: Snow, 25°F, low visibility
Result: Compare severity with same location in clear weather
```

### Use Case 3: Junction Safety
```
Location: Near major junction
Features: Junction, Traffic Signal, Crossing
Time: Any
Result: Assess accident severity risk at complex intersections
```

---

## 🔐 Privacy & Security

- No data is stored permanently
- All predictions are local
- API calls use only:
  - Public weather data (NOAA)
  - Public map data (OpenStreetMap)
- Your location data is not shared

---

## 📚 Additional Resources

- **Full Documentation**: See README.md
- **API Documentation**:
  - NOAA: https://www.ncdc.noaa.gov/cdo-web/webservices/v2
  - OpenStreetMap: https://wiki.openstreetmap.org/wiki/Overpass_API
- **Streamlit Docs**: https://docs.streamlit.io

---

## 🆘 Getting Help

1. Check this Quick Start Guide
2. Read the full README.md
3. Review troubleshooting section
4. Check API status if data fetching fails
5. Verify model files are present

---

## ✨ Pro Tips

- **Batch Testing**: Use manual input mode to quickly test multiple scenarios
- **Compare Conditions**: Try same location with different weather/time
- **Explore Patterns**: Use Dashboard to understand accident trends
- **Save Results**: Screenshot or note predictions for reference
- **API Limits**: If APIs are rate-limited, enable manual input mode

---

**Happy Predicting! 🎯**

Made with ❤️ using Streamlit, XGBoost, and Real-time APIs
