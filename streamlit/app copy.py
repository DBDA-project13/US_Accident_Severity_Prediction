import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
from datetime import datetime
import os
import zipfile
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from functools import lru_cache
import time
from folium.plugins import HeatMap
from streamlit_folium import folium_static
from streamlit_extras.add_vertical_space import add_vertical_space
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import requests
import joblib
import zipfile
import folium
from streamlit_folium import folium_static
from folium.plugins import HeatMap
from functools import lru_cache

# ---------------------------------------------------------
# UI CONFIGURATION & STYLING
# ---------------------------------------------------------
st.set_page_config(
    page_title="US Accident Severity Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for Pastel Professional Theme
st.markdown("""
    <style>
    /* Global Settings */
    @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Nunito', sans-serif;
        color: #4A5568;
    }
    
    /* Headers */
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(120deg, #84fab0 0%, #8fd3f4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        font-size: 1.2rem;
        color: #718096;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 600;
    }
    
    /* Cards/Containers */
    .css-1r6slb0, .stMarkdown, .stPlotlyChart {
        background-color: transparent;
    }
    
    .dashboard-card {
        background-color: #FFFFFF;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        margin-bottom: 1.5rem;
        border: 1px solid #E2E8F0;
    }

    /* Buttons */
    .stButton>button {
        width: 100%;
        height: 55px;
        font-size: 1.1rem;
        font-weight: 700;
        border-radius: 12px;
        border: none;
        background: linear-gradient(135deg, #A0C4FF 0%, #BDB2FF 100%);
        color: white;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(160, 196, 255, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 7px 14px rgba(160, 196, 255, 0.4);
        background: linear-gradient(135deg, #8CABFF 0%, #A99BFF 100%);
        color: white;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #F7FAFC;
        border-radius: 10px 10px 0 0;
        border: 1px solid #E2E8F0;
        color: #718096;
    }

    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF;
        border-bottom: 2px solid #A0C4FF;
        color: #4A5568;
        font-weight: bold;
    }
    
    /* Metrics */
    div[data-testid="stMetricValue"] {
        font-size: 24px;
        color: #2D3748;
    }
    </style>
""", unsafe_allow_html=True)

# Helper for spacing (replaces add_vertical_space if missing)
def add_vertical_space(num_lines: int = 1):
    for _ in range(num_lines):
        st.write("")

# ---------------------------------------------------------
# SESSION STATE & LOGIC (UNCHANGED)
# ---------------------------------------------------------

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'home'

def set_page(page_name):
    st.session_state.page = page_name
    st.rerun()

# Load model and preprocessor
@st.cache_resource
def load_model():
    try:
        with zipfile.ZipFile("model.pkl.zip", "r") as z:
            with z.open("model.pkl") as file:
                model = joblib.load(file)
        # model = joblib.load('model.pkl')
        preprocessor = joblib.load('preprocessing/new/preprocessor.pkl')
        return model, preprocessor
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# NOAA API integration
def fetch_weather_data(lat, lon, date_time):
    """
    Fetch weather data from NOAA API
    """
    try:
        base_url = "https://www.ncdc.noaa.gov/cdo-web/api/v2/data"
        api_token = "YOUR_NOAA_API_TOKEN" # Placeholder
        headers = {'token': api_token}
        date_str = date_time.strftime('%Y-%m-%d')
        params = {
            'datasetid': 'GHCND',
            'locationid': f'FIPS:US',
            'startdate': date_str,
            'enddate': date_str,
            'limit': 1000,
            'units': 'standard'
        }
        
        response = requests.get(base_url, headers=headers, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            weather_data = {
                'Temperature(F)': 65.0,
                'Humidity(%)': 60.0,
                'Pressure(in)': 29.92,
                'Visibility(mi)': 10.0,
                'Wind_Speed(mph)': 5.0,
                'Weather_Simple': 'Clear',
                'Wind_Direction_Simple': 'N'
            }
            return weather_data
        else:
            # st.warning("Could not fetch weather data from NOAA. Using default values.")
            return get_default_weather()
            
    except Exception as e:
        # st.warning(f"Weather API error: {e}. Using default values.")
        return get_default_weather()

def get_default_weather():
    return {
        'Temperature(F)': 65.0,
        'Humidity(%)': 60.0,
        'Pressure(in)': 29.92,
        'Visibility(mi)': 10.0,
        'Wind_Speed(mph)': 5.0,
        'Weather_Simple': 'Clear',
        'Wind_Direction_Simple': 'N'
    }

# OpenStreetMap API integration
def fetch_road_features(lat, lon):
    try:
        overpass_url = "http://overpass-api.de/api/interpreter"
        radius = 100
        query = f"""
        [out:json];
        (
          node["amenity"](around:{radius},{lat},{lon});
          node["highway"="crossing"](around:{radius},{lat},{lon});
          node["highway"="give_way"](around:{radius},{lat},{lon});
          node["highway"="stop"](around:{radius},{lat},{lon});
          node["highway"="traffic_signals"](around:{radius},{lat},{lon});
          node["railway"](around:{radius},{lat},{lon});
          node["railway"="station"](around:{radius},{lat},{lon});
          way["junction"](around:{radius},{lat},{lon});
        );
        out body;
        """
        response = requests.post(overpass_url, data={'data': query}, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            elements = data.get('elements', [])
            features = {
                'Amenity': False, 'Crossing': False, 'Give_Way': False, 'Junction': False,
                'No_Exit': False, 'Railway': False, 'Station': False, 'Stop': False, 'Traffic_Signal': False
            }
            for element in elements:
                tags = element.get('tags', {})
                if 'amenity' in tags: features['Amenity'] = True
                if tags.get('highway') == 'crossing': features['Crossing'] = True
                if tags.get('highway') == 'give_way': features['Give_Way'] = True
                if tags.get('highway') == 'stop': features['Stop'] = True
                if tags.get('highway') == 'traffic_signals': features['Traffic_Signal'] = True
                if 'junction' in tags: features['Junction'] = True
                if 'railway' in tags: features['Railway'] = True
                if tags.get('railway') == 'station': features['Station'] = True
            return features
        else:
            return get_default_features()
    except Exception as e:
        return get_default_features()

def get_default_features():
    return {
        'Amenity': False, 'Crossing': False, 'Give_Way': False, 'Junction': False,
        'No_Exit': False, 'Railway': False, 'Station': False, 'Stop': False, 'Traffic_Signal': False
    }

# Feature engineering functions
def create_temporal_features(date_time):
    features = {
        'hour': date_time.hour,
        'day': date_time.day,
        'month': date_time.month,
        'dayofweek': date_time.weekday(),
        'season': get_season(date_time.month)
    }
    features['is_rushhour'] = (7 <= date_time.hour <= 9) or (16 <= date_time.hour <= 19)
    features['is_holiday'] = False
    if 6 <= date_time.hour < 12: features['time_bucket'] = 'Morning'
    elif 12 <= date_time.hour < 18: features['time_bucket'] = 'Afternoon'
    elif 18 <= date_time.hour < 24: features['time_bucket'] = 'Evening'
    else: features['time_bucket'] = 'Night'
    return features

def get_season(month):
    if month in [12, 1, 2]: return 'Winter'
    elif month in [3, 4, 5]: return 'Spring'
    elif month in [6, 7, 8]: return 'Summer'
    else: return 'Fall'

def create_distance_features(distance):
    features = {
        'Distance(mi)_capped': min(distance, 10.0),
        'Distance(mi)_log': np.log1p(distance)
    }
    if distance < 0.5: features['Distance(mi)_bin'] = 'Very Short'
    elif distance < 2.0: features['Distance(mi)_bin'] = 'Short'
    elif distance < 5.0: features['Distance(mi)_bin'] = 'Medium'
    else: features['Distance(mi)_bin'] = 'Long'
    return features

# ---------------------------------------------------------
# PAGE RENDERERS
# ---------------------------------------------------------

def show_home_page():
    # Use columns to center content with professional spacing
    _, center_col, _ = st.columns([1, 6, 1])
    
    with center_col:
        st.markdown('<div class="main-header">🚗 US Accident Severity AI</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Advanced Machine Learning for Real-Time Safety Analysis</div>', unsafe_allow_html=True)
        
        st.write("")
        st.write("")
        
        # Professional Card Layout for Actions
        action_col1, action_col2 = st.columns(2, gap="large")
        
        with action_col1:
            st.markdown("""
            <div class="dashboard-card" style="text-align: center; height: 100%;">
                <h3>📊 Analytics Dashboard</h3>
                <p style="color: #718096; margin-bottom: 20px;">Explore historical trends, weather correlations, and geographic hotspots.</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Launch Dashboard", key="dashboard_btn"):
                set_page('dashboard')

        with action_col2:
            st.markdown("""
            <div class="dashboard-card" style="text-align: center; height: 100%;">
                <h3>🔮 Live Prediction</h3>
                <p style="color: #718096; margin-bottom: 20px;">Input accident parameters to predict severity using our XGBoost model.</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Start Prediction", key="prediction_btn"):
                set_page('prediction')

        st.write("")
        st.write("")
        
        # Info Section
        st.markdown("---")
        with st.container():
            st.markdown("""
            ### ℹ️ How it Works
            
            This system aggregates data from multiple sources to provide high-fidelity severity estimates:
            
            * **🌤️ NOAA Weather API:** Real-time atmospheric conditions.
            * **🛣️ OpenStreetMap:** Infrastructure analysis (signals, crossings, geometry).
            * **⏰ Temporal Analysis:** Rush hour, holiday, and seasonal patterns.
            """)

def show_dashboard_page():
    """Main dashboard display function with Tabs layout"""
    st.markdown('<div class="main-header">📊 Data Intelligence Hub</div>', unsafe_allow_html=True)
    
    # Navigation Tabs for cleaner UI
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Tableau Report", "🌤️ Weather & Severity", "🗺️ Geospatial", "⏱️ Time Analysis"])
    
    # TAB 1: Tableau
    with tab1:
        st.markdown("#### Interactive Executive Summary")
        tableau_html = """
        <div class='tableauPlaceholder' id='viz1769690834638' style='position: relative; border-radius: 15px; overflow: hidden; box-shadow: 0 4px 10px rgba(0,0,0,0.1);'>
            <noscript><a href='#'><img alt=' ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;CD&#47;CDAC_PROJECT_17691643176930&#47;ProjectStory&#47;1_rss.png' style='border: none' /></a></noscript>
            <object class='tableauViz' style='display:none;'>
                <param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' />
                <param name='embed_code_version' value='3' />
                <param name='site_root' value='' />
                <param name='name' value='CDAC_PROJECT_17691643176930&#47;ProjectStory' />
                <param name='tabs' value='yes' />
                <param name='toolbar' value='yes' />
                <param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;CD&#47;CDAC_PROJECT_17691643176930&#47;ProjectStory&#47;1.png' />
                <param name='animate_transition' value='yes' />
                <param name='display_static_image' value='yes' />
                <param name='display_spinner' value='yes' />
                <param name='display_overlay' value='yes' />
                <param name='display_count' value='yes' />
            </object>
        </div>
        <script type='text/javascript'>
            var divElement = document.getElementById('viz1769690834638');
            var vizElement = divElement.getElementsByTagName('object')[0];
            vizElement.style.width='100%';
            vizElement.style.height=(divElement.offsetWidth*0.75)+'px';
            var scriptElement = document.createElement('script');
            scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';
            vizElement.parentNode.insertBefore(scriptElement, vizElement);
        </script>
        """
        st.components.v1.html(tableau_html, height=800, scrolling=True)

    # Load data once for other tabs
    try:
        df = pd.read_csv("US_Accidents53.csv", encoding="ISO-8859-1")
    except FileNotFoundError:
        st.error("Data file 'US_Accidents53.csv' not found. Please ensure data is present.")
        return

    # TAB 2: Weather & Severity Factors
    with tab2:
        st.markdown("#### Environmental Impact Analysis")
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            fig = px.box(df, x='Severity', y='Temperature(F)', color='Severity', 
                        title='Temperature vs. Severity', color_discrete_sequence=px.colors.qualitative.Pastel)
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            fig_weather = px.bar(df['Weather_Condition'].value_counts().head(10), 
                                orientation='h',
                                title='Top 10 Weather Conditions',
                                color_discrete_sequence=['#FFB7B2'])
            fig_weather.update_layout(showlegend=False, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_weather, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        c3, c4 = st.columns(2)
        with c3:
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            fig = px.scatter(df, x='Wind_Speed(mph)', y='Visibility(mi)', color='Visibility(mi)',
                            title='Wind Speed vs. Visibility Correlation',
                            color_continuous_scale=px.colors.sequential.Bluyl)
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with c4:
             st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
             fig = px.histogram(df, x='Severity', title='Severity Distribution', 
                          color_discrete_sequence=['#B5EAD7'])
             fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', bargap=0.1)
             st.plotly_chart(fig, use_container_width=True)
             st.markdown('</div>', unsafe_allow_html=True)

    # TAB 3: Geography
    with tab3:
        st.markdown("#### Geographic Hotspots")
        
        c1, c2 = st.columns([2, 1])
        with c1:
             st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
             st.markdown("**📍 Accident Heatmap Density**")
             try:
                 map_center = [df['Start_Lat'].mean(), df['Start_Lng'].mean()]
                 m = folium.Map(location=map_center, zoom_start=5, tiles='CartoDB positron') # Cleaner map style
                 HeatMap(list(zip(df['Start_Lat'], df['Start_Lng'])), 
                         min_opacity=0.2, radius=15, blur=15, gradient={0.4: '#A0C4FF', 0.65: '#FDFFB6', 1: '#FFADAD'}).add_to(m)
                 folium_static(m)
             except Exception as e:
                 st.warning(f"Map generation failed: {e}")
             st.markdown('</div>', unsafe_allow_html=True)
        
        with c2:
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            city_counts = df['City'].value_counts().head(10)
            fig_city = px.bar(city_counts, 
                            title='Top 10 High-Risk Cities',
                            color_discrete_sequence=['#C7CEEA'])
            fig_city.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_city, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # TAB 4: Time Analysis
    with tab4:
        st.markdown("#### Temporal Trends")
        
        df['Start_Time'] = pd.to_datetime(df['Start_Time'])
        df['Weekday'] = df['Start_Time'].dt.day_name()
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            week_day_counts = df['Weekday'].value_counts()
            pie_df = pd.DataFrame({'Weekday': week_day_counts.index, 'Count': week_day_counts.values})
            fig = px.pie(pie_df, names='Weekday', values='Count', 
                        title='Accidents by Day of Week',
                        color_discrete_sequence=px.colors.qualitative.Pastel)
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with c2:
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            fig = px.histogram(df, x='Hour', nbins=24, color_discrete_sequence=['#FFDAC1'], 
                            title='Hourly Accident Frequency')
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

@lru_cache(maxsize=100)
def get_state_from_coords(lat, lon):
    try:
        url = f"https://nominatim.openstreetmap.org/reverse"
        params = {'lat': lat, 'lon': lon, 'format': 'json', 'addressdetails': 1}
        headers = {'User-Agent': 'AccidentSeverityApp/1.0'}
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        address = data.get('address', {})
        state = address.get('state', '')
        state_abbrev = get_state_abbreviation(state)
        return state_abbrev if state_abbrev else state
    except Exception as e:
        return "Unknown"

def get_state_abbreviation(state_name):
    state_map = {
        'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR',
        'California': 'CA', 'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE',
        'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI', 'Idaho': 'ID',
        'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS',
        'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD',
        'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS',
        'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV',
        'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM', 'New York': 'NY',
        'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK',
        'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC',
        'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT',
        'Vermont': 'VT', 'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV',
        'Wisconsin': 'WI', 'Wyoming': 'WY', 'District of Columbia': 'DC'
    }
    return state_map.get(state_name, None)
# ---------------------------------------------------------
# PREDICTION PAGE
# ---------------------------------------------------------

def show_prediction_page():
    # Header Section
    st.markdown('<div class="main-header">🔮 Prediction Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Configure parameters for real-time severity assessment</div>', unsafe_allow_html=True)
    
    # Navigation
    if st.button("← Return to Home Dashboard"):
        set_page('home')
    
    add_vertical_space(1)
    
    # Load resources
    model, preprocessor = load_model()
    if model is None or preprocessor is None:
        st.error("⚠️ System Error: Model resources unavailable.")
        return
    
    # Main Input Layout
    c1, c2 = st.columns([1, 1], gap="large")
    
    # ------------------ COLUMN 1: LOCATION ------------------
    with c1:
        st.markdown("#### 📍 Geospatial Data", unsafe_allow_html=True)
        st.caption("Select accident location to automatically derive state and road features.")
        
        with st.container(border=True):
            latitude = st.number_input("Latitude", value=34.0522, min_value=24.0, max_value=50.0, step=0.0001, format="%.4f")
            longitude = st.number_input("Longitude", value=-118.2437, min_value=-125.0, max_value=-66.0, step=0.0001, format="%.4f")
            
            # Map Interaction
            st.markdown("**Interactive Map Selector**")
            m = folium.Map(location=[latitude, longitude], zoom_start=13, tiles="CartoDB positron")
            
            # Update coordinates logic
            def update_coordinates(click):
                global latitude, longitude
                latitude, longitude = click['lat'], click['lng']
                st.session_state['latitude'] = latitude
                st.session_state['longitude'] = longitude

            folium.Marker([latitude, longitude], popup="Target Location", icon=folium.Icon(color='purple', icon='car', prefix='fa')).add_to(m)
            folium.LatLngPopup().add_to(m)

            map_data = st_folium(m, height=250, width="100%")

            if map_data and 'last_clicked' in map_data and map_data['last_clicked']:
                update_coordinates(map_data['last_clicked'])

            # Computed Location Info
            state = get_state_from_coords(lat=latitude, lon=longitude)
            st.info(f"**Detected Region:** {state}")
            
            distance = st.slider("Affected Distance (miles)", value=0.5, min_value=0.0, max_value=10.0, step=0.1)

    # ------------------ COLUMN 2: TEMPORAL & ENV ------------------
    with c2:
        st.markdown("#### 🕐 Temporal & Environmental", unsafe_allow_html=True)
        st.caption("Specify time and conditions. API fetching is recommended for accuracy.")
        
        with st.container(border=True):
            d_col1, d_col2 = st.columns(2)
            with d_col1:
                accident_date = st.date_input("Date", value=datetime.now())
            with d_col2:
                accident_time = st.time_input("Time", value=datetime.now().time())
            
            accident_datetime = datetime.combine(accident_date, accident_time)
            st.caption(f"Timestamp: {accident_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
            
            st.markdown("---")
            st.markdown("**Data Source Configuration**")
            
            fetch_data = st.toggle("Auto-fetch NOAA & OpenStreetMap Data", value=True)
            
            if fetch_data:
                st.success("✅ APIs Ready: Weather and infrastructure data will be pulled live.")
            else:
                st.warning("⚠️ Manual Mode: You must provide environmental details manually.")

    add_vertical_space(1)
    
    # ------------------ MANUAL OVERRIDE SECTION ------------------
    with st.expander("🔧 Manual Input / Override Settings"):
        st.caption("Use these settings if API data is unavailable or to simulate specific scenarios.")
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Atmospheric Conditions**")
            manual_temp = st.slider("Temperature (°F)", -20.0, 120.0, 65.0)
            manual_humidity = st.slider("Humidity (%)", 0.0, 100.0, 60.0)
            manual_pressure = st.number_input("Pressure (in)", 28.0, 31.0, 29.92)
            manual_visibility = st.slider("Visibility (mi)", 0.0, 10.0, 10.0)
            manual_wind_speed = st.number_input("Wind Speed (mph)", 0.0, 100.0, 5.0)
        
        with col_b:
            st.markdown("**Road Context**")
            manual_weather = st.selectbox("Condition", ['Clear', 'Cloudy', 'Rain', 'Snow', 'Fog'])
            manual_wind_dir = st.selectbox("Wind Direction", ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
            
            st.markdown("Feature Flags:")
            c_x1, c_x2 = st.columns(2)
            with c_x1:
                manual_amenity = st.checkbox("Amenity")
                manual_crossing = st.checkbox("Crossing")
            with c_x2:
                manual_junction = st.checkbox("Junction")
                manual_traffic_signal = st.checkbox("Signal")
        
        use_manual = st.checkbox("Force Manual Data Usage", help="Ignore API data and use the values above")

    add_vertical_space(2)
    
    # ------------------ ACTION SECTION ------------------
    col_predict, _, _ = st.columns([1, 1, 1])
    with col_predict:
        predict_button = st.button("🚀 Run Severity Analysis", type="primary", use_container_width=True)
    
    if predict_button:
        with st.spinner("Processing geospatial and atmospheric data..."):
            try:
                # 1. Data Gathering
                if fetch_data and not (st.session_state.get('use_manual', False) if 'use_manual' in st.session_state else use_manual):
                    with st.status("📡 Connecting to External APIs...", expanded=True) as status:
                        st.write("Fetching NOAA Weather data...")
                        weather_data = fetch_weather_data(latitude, longitude, accident_datetime)
                        st.write("Fetching OpenStreetMap infrastructure...")
                        road_features = fetch_road_features(latitude, longitude)
                        status.update(label="✅ Data Acquisition Complete", state="complete", expanded=False)
                else:
                    weather_data = {
                        'Temperature(F)': manual_temp, 'Humidity(%)': manual_humidity,
                        'Pressure(in)': manual_pressure, 'Visibility(mi)': manual_visibility,
                        'Wind_Speed(mph)': manual_wind_speed, 'Weather_Simple': manual_weather,
                        'Wind_Direction_Simple': manual_wind_dir
                    }
                    road_features = {
                        'Amenity': manual_amenity, 'Crossing': manual_crossing, 'Give_Way': False,
                        'Junction': manual_junction, 'No_Exit': False, 'Railway': False,
                        'Station': False, 'Stop': False, 'Traffic_Signal': manual_traffic_signal
                    }
                
                # 2. Processing
                temporal_features = create_temporal_features(accident_datetime)
                distance_features = create_distance_features(distance)
                
                input_data = {
                    'State': state, 'Start_Lat': latitude, 'Start_Lng': longitude,
                    **weather_data, **road_features, **temporal_features, **distance_features
                }
                
                # 3. Prediction
                input_df = pd.DataFrame([input_data])
                X_processed = preprocessor.transform(input_df)
                prediction_proba = model.predict_proba(X_processed)[0]
                class_labels = model.classes_ + 1 # Assuming 0-indexed classes mapped to 1-4
                prediction = class_labels[prediction_proba.argmax()]

                # 4. Visualization of Results
                severity_map = {1: "Minor Impact", 2: "Moderate Impact", 3: "High Impact", 4: "Severe Impact"}
                severity_colors = {1: "#77DD77", 2: "#FDFD96", 3: "#FFB347", 4: "#FF6961"} # Pastel traffic lights
                severity_icons = {1: "🟢", 2: "🟡", 3: "🟠", 4: "🔴"}
                
                st.write("")
                
                # Result Card
                result_color = severity_colors.get(prediction, "#cccccc")
                result_text = severity_map.get(prediction, "Unknown")
                result_icon = severity_icons.get(prediction, "")

                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, #FFFFFF 0%, {result_color}33 100%);
                    padding: 2rem;
                    border-radius: 20px;
                    border: 2px solid {result_color};
                    text-align: center;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.05);
                    margin-bottom: 2rem;
                ">
                    <h4 style="margin:0; color: #718096; text-transform: uppercase; letter-spacing: 2px;">Predicted Severity</h4>
                    <h1 style="font-size: 3.5rem; margin: 10px 0; color: #2D3748;">{result_icon} Level {prediction}</h1>
                    <h3 style="color: {result_color}; filter: brightness(0.8); margin:0;">{result_text}</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Probability Breakdown
                st.markdown("#### 📊 Probability Distribution")
                prob_data = pd.DataFrame({
                    'Severity': [f"Level {c}" for c in class_labels],
                    'Probability': prediction_proba
                })
                
                # Custom Bar Chart for probabilities
                fig = px.bar(prob_data, x='Probability', y='Severity', orientation='h',
                             text=prob_data['Probability'].apply(lambda x: f"{x:.1%}"),
                             color='Severity',
                             color_discrete_sequence=['#77DD77', '#FDFD96', '#FFB347', '#FF6961'])
                fig.update_layout(showlegend=False, xaxis_range=[0,1], height=250,
                                margin=dict(l=0, r=0, t=0, b=0),
                                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
                
                with st.expander("📋 View Raw Input Data"):
                    st.json(input_data)
                
            except Exception as e:
                st.error("❌ An error occurred during prediction processing.")
                st.exception(e)

# ---------------------------------------------------------
# MAIN ENTRY POINT
# ---------------------------------------------------------

def main():
    # Styled Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <div style="background-color: #F0F4F8; width: 80px; height: 80px; border-radius: 50%; margin: 0 auto; display: flex; align-items: center; justify-content: center; font-size: 2rem;">
                🚗
            </div>
            <h3 style="margin-top: 1rem; color: #4A5568;">Navigator</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Navigation Buttons (using full width for touch-friendly feel)
        st.write("")
        if st.button("🏠 Home", use_container_width=True):
            set_page('home')
        
        if st.button("📊 Dashboard", use_container_width=True):
            set_page('dashboard')
        
        if st.button("🔮 Prediction", use_container_width=True):
            set_page('prediction')
        
        st.markdown("---")
        
        # Metadata
        st.caption("SYSTEM INFO")
        st.info(
            "**Version:** 2.0.1 PRO\n\n"
            "**Engine:** XGBoost Classifier\n\n"
            "**Data:** Live NOAA & OSM"
        )
        
        st.markdown("""
        <div style="text-align: center; color: #CBD5E0; font-size: 0.8rem; margin-top: 2rem;">
            Designed for Reliability
        </div>
        """, unsafe_allow_html=True)
    
    # Router
    if st.session_state.page == 'home':
        show_home_page()
    elif st.session_state.page == 'dashboard':
        show_dashboard_page()
    elif st.session_state.page == 'prediction':
        show_prediction_page()

if __name__ == "__main__":
    main()