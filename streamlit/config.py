# API Configuration File
# Copy this file to config.py and add your actual API keys

# NOAA API Configuration
# Get your free API token from: https://www.ncdc.noaa.gov/cdo-web/token
NOAA_API_TOKEN = "hAdvYrkmAxasDTDiotuVaiTkIxalSSMm"

# OpenStreetMap Configuration
# OpenStreetMap Overpass API doesn't require authentication
# But you can configure the endpoint and timeout
OSM_OVERPASS_URL = "http://overpass-api.de/api/interpreter"
OSM_TIMEOUT = 30  # seconds
OSM_SEARCH_RADIUS = 100  # meters

# Model Configuration
MODEL_PATH = "model.pkl"
PREPROCESSOR_PATH = "preprocessing/new/preprocessor.pkl"

# Feature Configuration
NUMERIC_FEATURES = [
    'Start_Lat', 'Start_Lng', 'Temperature(F)', 'Humidity(%)', 
    'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)',
    'hour', 'day', 'month', 'dayofweek', 
    'Distance(mi)_capped', 'Distance(mi)_log'
]

BOOL_FEATURES = [
    'Amenity', 'Crossing', 'Give_Way', 'Junction', 'No_Exit', 
    'Railway', 'Station', 'Stop', 'Traffic_Signal', 
    'is_rushhour', 'is_holiday'
]

CATEGORICAL_FEATURES = [
    'State', 'Weather_Simple', 'Wind_Direction_Simple', 
    'time_bucket', 'Distance(mi)_bin', 'season'
]

# Severity Mapping
SEVERITY_LEVELS = {
    1: {
        'name': 'Low',
        'icon': '🟢',
        'description': 'Minor accident with minimal impact on traffic flow',
        'color': '#28a745'
    },
    2: {
        'name': 'Moderate',
        'icon': '🟡',
        'description': 'Moderate accident with some traffic disruption',
        'color': '#ffc107'
    },
    3: {
        'name': 'High',
        'icon': '🟠',
        'description': 'Serious accident causing significant traffic delays',
        'color': '#fd7e14'
    },
    4: {
        'name': 'Severe',
        'icon': '🔴',
        'description': 'Critical accident with major traffic impact',
        'color': '#dc3545'
    }
}
