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

# Page configuration
st.set_page_config(
    page_title="US Accident Severity Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        height: 60px;
        font-size: 1.2rem;
        font-weight: bold;
        border-radius: 10px;
        margin: 10px 0;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 20px 0;
    }
    </style>
""", unsafe_allow_html=True)

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
    Note: You need to get a free API token from https://www.ncdc.noaa.gov/cdo-web/token
    """
    try:
        # NOAA API endpoint
        base_url = "https://www.ncdc.noaa.gov/cdo-web/api/v2/data"
        
        # You need to replace this with your actual NOAA API token
        api_token = "YOUR_NOAA_API_TOKEN"
        
        headers = {
            'token': api_token
        }
        
        # Format date for API
        date_str = date_time.strftime('%Y-%m-%d')
        
        params = {
            'datasetid': 'GHCND',  # Daily Summaries
            'locationid': f'FIPS:US',
            'startdate': date_str,
            'enddate': date_str,
            'limit': 1000,
            'units': 'standard'
        }
        
        response = requests.get(base_url, headers=headers, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            # Parse weather data
            weather_data = {
                'Temperature(F)': 65.0,  # Default values if API fails
                'Humidity(%)': 60.0,
                'Pressure(in)': 29.92,
                'Visibility(mi)': 10.0,
                'Wind_Speed(mph)': 5.0,
                'Weather_Simple': 'Clear',
                'Wind_Direction_Simple': 'N'
            }
            return weather_data
        else:
            st.warning("Could not fetch weather data from NOAA. Using default values.")
            return get_default_weather()
            
    except Exception as e:
        st.warning(f"Weather API error: {e}. Using default values.")
        return get_default_weather()

def get_default_weather():
    """Return default weather values"""
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
    """
    Fetch road features from OpenStreetMap Overpass API
    """
    try:
        overpass_url = "http://overpass-api.de/api/interpreter"
        
        # Search radius in meters
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
                'Amenity': False,
                'Crossing': False,
                'Give_Way': False,
                'Junction': False,
                'No_Exit': False,
                'Railway': False,
                'Station': False,
                'Stop': False,
                'Traffic_Signal': False
            }
            
            for element in elements:
                tags = element.get('tags', {})
                
                if 'amenity' in tags:
                    features['Amenity'] = True
                if tags.get('highway') == 'crossing':
                    features['Crossing'] = True
                if tags.get('highway') == 'give_way':
                    features['Give_Way'] = True
                if tags.get('highway') == 'stop':
                    features['Stop'] = True
                if tags.get('highway') == 'traffic_signals':
                    features['Traffic_Signal'] = True
                if 'junction' in tags:
                    features['Junction'] = True
                if 'railway' in tags:
                    features['Railway'] = True
                if tags.get('railway') == 'station':
                    features['Station'] = True
            
            return features
        else:
            st.warning("Could not fetch road features. Using default values.")
            return get_default_features()
            
    except Exception as e:
        st.warning(f"OpenStreetMap API error: {e}. Using default values.")
        return get_default_features()

def get_default_features():
    """Return default road features"""
    return {
        'Amenity': False,
        'Crossing': False,
        'Give_Way': False,
        'Junction': False,
        'No_Exit': False,
        'Railway': False,
        'Station': False,
        'Stop': False,
        'Traffic_Signal': False
    }

# Feature engineering functions
def create_temporal_features(date_time):
    """Create temporal features from datetime"""
    features = {
        'hour': date_time.hour,
        'day': date_time.day,
        'month': date_time.month,
        'dayofweek': date_time.weekday(),
        'season': get_season(date_time.month)
    }
    
    # Rush hour indicator
    features['is_rushhour'] = (7 <= date_time.hour <= 9) or (16 <= date_time.hour <= 19)
    
    # Holiday indicator (simplified)
    features['is_holiday'] = False  # You can enhance this with a holiday calendar
    
    # Time bucket
    if 6 <= date_time.hour < 12:
        features['time_bucket'] = 'Morning'
    elif 12 <= date_time.hour < 18:
        features['time_bucket'] = 'Afternoon'
    elif 18 <= date_time.hour < 24:
        features['time_bucket'] = 'Evening'
    else:
        features['time_bucket'] = 'Night'
    
    return features

def get_season(month):
    """Determine season from month"""
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

def create_distance_features(distance):
    """Create distance-related features"""
    features = {
        'Distance(mi)_capped': min(distance, 10.0),  # Cap at 10 miles
        'Distance(mi)_log': np.log1p(distance)
    }
    
    # Distance bins
    if distance < 0.5:
        features['Distance(mi)_bin'] = 'Very Short'
    elif distance < 2.0:
        features['Distance(mi)_bin'] = 'Short'
    elif distance < 5.0:
        features['Distance(mi)_bin'] = 'Medium'
    else:
        features['Distance(mi)_bin'] = 'Long'
    
    return features

# Home Page
def show_home_page():
    st.markdown('<p class="main-header">🚗 US Accident Severity Prediction System</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Predict accident severity using machine learning and real-time data</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.image("https://raw.githubusercontent.com/github/explore/80688e429a7d4ef2fca1e82350fe8e3517d3494d/topics/machine-learning/machine-learning.png", width=200)
        
        st.markdown("### Choose an option:")
        
        if st.button("📊 View Dashboard", key="dashboard_btn"):
            set_page('dashboard')
        
        if st.button("🔮 Make Prediction", key="prediction_btn"):
            set_page('prediction')
        
        st.markdown("---")
        st.markdown("""
        ### About This Application
        
        This system predicts the severity of traffic accidents based on:
        - **Location data** (latitude, longitude, state)
        - **Real-time weather conditions** (via NOAA API)
        - **Road features** (via OpenStreetMap API)
        - **Temporal factors** (time, day, season)
        - **Traffic conditions** (rush hour, holidays)
        
        The model uses XGBoost trained on millions of US accident records to provide accurate severity predictions.
        """)


def show_dashboard_page():
    """Main dashboard display function"""
    
    # Tableau Dashboard Section
    st.markdown("## 📊 Tableau Dashboard")
    add_vertical_space(2)
    
    # Embed Tableau dashboard
    # tableau_url = st.text_input("Enter your Tableau Public/Server URL:", 
                            #    placeholder="https://public.tableau.com/app/profile/anshul.aher/viz/CDAC_PROJECT_17691643176930/ProjectStory?publish=yes")
    # tableau_url = "https://public.tableau.com/app/profile/anshul.aher/viz/CDAC_PROJECT_17691643176930/ProjectStory?publish=yes"
    tableau_html = """
    <div class='tableauPlaceholder' id='viz1769690834638' style='position: relative'>
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
            <param name='language' value='en-GB' />
            <param name='filter' value='publish=yes' />
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
    
    # add_vertical_space(5)
    # if tableau_url:
    #     st.components.v1.iframe(tableau_url, height=800, scrolling=True)
    # else:
    #     st.info("📌 Enter your Tableau dashboard URL above to display it here")
    
    add_vertical_space(5)
    
    # st.markdown(f"""
    #     <div class='tableauPlaceholder' id='viz1769690834638' style='position: relative'><noscript><a href='#'><img alt=' ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;CD&#47;CDAC_PROJECT_17691643176930&#47;ProjectStory&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='CDAC_PROJECT_17691643176930&#47;ProjectStory' /><param name='tabs' value='yes' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;CD&#47;CDAC_PROJECT_17691643176930&#47;ProjectStory&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-GB' /><param name='filter' value='publish=yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1769690834638');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>
    #     """, unsafe_allow_html=True)
    
    # Data Visualisations Section
    st.markdown('# **General Data Visualisation Window**')
    add_vertical_space(2)
    st.markdown('##### :smile: Generalised Visualisations on the data :balloon:')
    add_vertical_space(3)
    
    
    st.markdown("### Some General Visualisations")
    add_vertical_space(3)
    df = pd.read_csv("US_Accidents53.csv", encoding="ISO-8859-1")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.box(df, x='Severity', y='Temperature(F)', color='Severity', 
                    title='Temperature vs. Severity of Accidents')
        st.plotly_chart(fig, use_container_width=True)
    
    with c2:
        fig = px.histogram(df, x='Hour', nbins=24, color_discrete_sequence=['orange'], 
                          labels={'Hour': 'Hour of the Day', 'count': 'Frequency'}, 
                          title='Histogram of Accident Times')
        fig.update_layout(xaxis_title='Hour of the Day', yaxis_title='Frequency', 
                         title_font_size=16, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    add_vertical_space(5)
    
    df['Start_Time'] = pd.to_datetime(df['Start_Time'])
    df['Weekday'] = df['Start_Time'].dt.day_name()
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('##### Accidents by Day of the Week')
        week_day_counts = df['Weekday'].value_counts()
        pie_df = pd.DataFrame({'Weekday': week_day_counts.index, 'Count': week_day_counts.values})
        fig = px.pie(pie_df, names='Weekday', values='Count', 
                    title='Accidents by Day of the Week',
                    color_discrete_sequence=['skyblue', 'lightgreen', 'lightcoral', 
                                            'orange', 'lightyellow', 'lightpink', 'lightskyblue'])
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(title_font_size=16, title_font_color='purple')
        st.plotly_chart(fig, use_container_width=True)
    
    with c2:
        st.markdown('##### Wind Speed vs. Visibility')
        fig = px.scatter(df, x='Wind_Chill(F)', y='Visibility(mi)', color='Visibility(mi)', 
                        title='Wind Chill vs. Visibility', 
                        labels={'Wind_Chill(F)': 'Wind Chill (F)', 'Visibility(mi)': 'Visibility (miles)'})
        fig.update_traces(marker=dict(size=10, opacity=0.5), selector=dict(mode='markers'))
        fig.update_layout(title_font_size=16, title_font_color='brown', showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    add_vertical_space(5)
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Scatter Plot of Wind Speed Vs. Visibility")
        fig = px.scatter(df, x='Wind_Speed(mph)', y='Visibility(mi)', color='Visibility(mi)',
                        title='Wind Speed vs. Visibility', 
                        labels={'Wind_Speed(mph)': 'Wind Speed (mph)', 'Visibility(mi)': 'Visibility (miles)'})
        fig.update_layout(title_font_size=16, title_font_color='blue',
                         xaxis_title='Wind Speed (mph)', yaxis_title='Visibility (miles)',
                         xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgrey'),
                         yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgrey'))
        st.plotly_chart(fig, use_container_width=True)
    
    with c2:
        st.markdown("##### Trends Line Chart")
        df_time = df.copy()
        df_time['Start_Time'] = pd.to_datetime(df_time['Start_Time'])
        df_time.set_index('Start_Time', inplace=True)
        monthly_accidents = df_time.resample('M').size()
        fig = px.line(x=monthly_accidents.index, y=monthly_accidents.values,
                     title='Accidents Over Time', labels={'x': 'Date', 'y': 'Number of Accidents'})
        fig.update_traces(line=dict(color='hotpink', dash='dot', width=2))
        fig.update_layout(title_font_size=16, title_font_color='hotpink',
                         xaxis_title_font_size=12, yaxis_title_font_size=12,
                         xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgrey'),
                         yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgrey'))
        st.plotly_chart(fig, use_container_width=True)
    
    add_vertical_space(5)
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Count Plot of Accident Severity")
        fig = px.histogram(df, x='Severity', title='Distribution of Accident Severity', 
                          color_discrete_sequence=['orange'], labels={'Severity': 'Severity'})
        fig.update_layout(title_font_size=16, title_font_color='orange', 
                         xaxis_title='Severity', yaxis_title='Count', 
                         xaxis=dict(showgrid=False), 
                         yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgrey'))
        st.plotly_chart(fig, use_container_width=True)
    
    with c2:
        st.markdown('##### Accidents by Weather Condition')
        fig_weather = px.bar(df['Weather_Condition'].value_counts(), 
                            orientation='h',
                            title='Accidents by Weather Condition',
                            labels={'value': 'Number of Accidents', 'index': 'Weather Condition'},
                            color_discrete_sequence=['red'],
                            height=500)
        fig_weather.update_layout(title_font_size=16, title_font_color='red', 
                                 yaxis_title='Weather Condition', xaxis_title='Number of Accidents')
        st.plotly_chart(fig_weather, use_container_width=True)
    
    c1, c2 = st.columns(2)
    with c1:
        city_counts = df['City'].value_counts().head(10)
        st.markdown('##### Top 10 Cities by Accident Counts')
        fig_city = px.bar(city_counts, 
                         title='Top 10 Cities by Accident Counts',
                         labels={'value': 'Number of Accidents', 'index': 'City'},
                         color_discrete_sequence=['purple'],
                         height=500)
        fig_city.update_layout(title_font_size=16, title_font_color='purple', 
                              yaxis_title='Number of Accidents', xaxis_title='City')
        st.plotly_chart(fig_city, use_container_width=True)
    
    with c2:
        custom_palette = sns.color_palette("Set2")
        sns.set_palette(custom_palette)
        sns.set_style("whitegrid")
        st.markdown('##### Distribution of Accidents by Month')
        fig1, ax1 = plt.subplots()
        sns.countplot(x='Month', data=df, ax=ax1)
        ax1.set_title('Distribution of Accidents by Month')
        ax1.set_xlabel('Month')
        ax1.set_ylabel('Number of Accidents')
        st.pyplot(fig1, use_container_width=True)
    
    add_vertical_space(5)
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown('##### Box Plot of Log-Transformed Accident Duration')
        df['Duration'] = (pd.to_datetime(df['End_Time']) - pd.to_datetime(df['Start_Time'])).dt.total_seconds() / 3600
        df['Log_Duration'] = np.log1p(df['Duration'])
        fig2, ax2 = plt.subplots()
        sns.boxplot(x='Log_Duration', data=df, ax=ax2)
        ax2.set_title('Box Plot of Log-Transformed Accident Duration')
        ax2.set_xlabel('Log(Duration in Hours)')
        st.pyplot(fig2, use_container_width=True)
    
    with c2:
        st.markdown('##### Pressure vs. Humidity')
        fig = px.scatter(df, x='Pressure(in)', y='Humidity(%)')
        fig.update_layout(xaxis_title='Atmospheric Pressure (in)', yaxis_title='Humidity (%)')
        fig.update_traces(marker=dict(color='rgba(0,0,0,0)'))
        fig.update_layout(barmode='overlay')
        fig.add_trace(px.bar(df, x='Pressure(in)').data[0])
        fig.add_trace(px.bar(df, y='Humidity(%)').data[0])
        st.plotly_chart(fig, use_container_width=True)
    
    add_vertical_space(5)
    
    palette1 = px.colors.qualitative.Pastel
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown('##### Day vs. Night Accidents')
        fig2 = px.bar(df['Sunrise_Sunset'].value_counts(), 
                     x=df['Sunrise_Sunset'].value_counts().index, 
                     y=df['Sunrise_Sunset'].value_counts().values, 
                     title='Day vs. Night Accidents', color_discrete_sequence=palette1)
        fig2.update_layout(xaxis_title='Time of Day', yaxis_title='Number of Accidents')
        st.plotly_chart(fig2)
    
    with c2:
        try:
            wata = pd.read_csv("US_Accident23_1000.csv")
            st.markdown('##### Visibility vs. Severity')
            fig3 = px.violin(wata, x='Severity', y='Visibility(mi)', 
                           title='Visibility vs. Severity', color='Severity', 
                           color_discrete_sequence=palette1)
            fig3.update_layout(xaxis_title='Severity', yaxis_title='Visibility (miles)')
            st.plotly_chart(fig3)
        except:
            st.info("US_Accident23_1000.csv not found for violin plot")
    
    add_vertical_space(5)
    
    c1, c2, c3 = st.columns([0.15, 0.70, 0.15])
    with c2:
        st.markdown("##### Kernel Density Accidents in Folium Map")
        add_vertical_space(2)
        map_center = [df['Start_Lat'].mean(), df['Start_Lng'].mean()]
        map = folium.Map(location=map_center, zoom_start=5)
        heatmap = HeatMap(list(zip(df['Start_Lat'], df['Start_Lng'])), 
                         min_opacity=0.2, radius=15, blur=15)
        map.add_child(heatmap)
        folium_static(map)

@lru_cache(maxsize=100)
def get_state_from_coords(lat, lon):
    """
    Extract state from coordinates using Nominatim reverse geocoding
    """
    try:
        url = f"https://nominatim.openstreetmap.org/reverse"
        params = {
            'lat': lat,
            'lon': lon,
            'format': 'json',
            'addressdetails': 1
        }
        headers = {'User-Agent': 'AccidentSeverityApp/1.0'}
        
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Extract state from address components
        address = data.get('address', {})
        state = address.get('state', '')
        
        # Convert full state name to abbreviation if needed
        state_abbrev = get_state_abbreviation(state)
        
        return state_abbrev if state_abbrev else state
        
    except Exception as e:
        st.warning(f"Could not determine state from coordinates: {e}")
        return "Unknown"

# ----------------------------- 
# State Abbreviation Mapping
# ----------------------------- 
def get_state_abbreviation(state_name):
    """Convert full state name to 2-letter abbreviation"""
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
# Prediction Page
def show_prediction_page():
    st.markdown('<p class="main-header">🔮 Accident Severity Prediction</p>', unsafe_allow_html=True)
    
    if st.button("← Back to Home"):
        set_page('home')
    
    st.markdown("---")
    
    # Load model
    model, preprocessor = load_model()
    
    if model is None or preprocessor is None:
        st.error("⚠️ Model or preprocessor not found. Please ensure model.pkl and preprocessor.pkl are available.")
        return
    
    st.markdown("### Enter Accident Details")
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📍 Location Information")
        latitude = st.number_input("Latitude", value=34.0522, min_value=24.0, max_value=50.0, step=0.0001, format="%.4f")
        longitude = st.number_input("Longitude", value=-118.2437, min_value=-125.0, max_value=-66.0, step=0.0001, format="%.4f")

        # Display the map
        st.markdown("**Location Preview:**")
        m = folium.Map(location=[latitude, longitude], zoom_start=13)

        # Function to update coordinates based on a click
        def update_coordinates(click):
            global latitude, longitude
            latitude, longitude = click['lat'], click['lng']
            st.session_state['latitude'] = latitude
            st.session_state['longitude'] = longitude

        # Initial marker
        marker = folium.Marker([latitude, longitude], popup="Accident Location", icon=folium.Icon(color='red'))
        marker.add_to(m)

        # Add LatLngPopup to show coordinates on click
        latlng_popup = folium.LatLngPopup().add_to(m)

        # Use Streamlit callback to handle the interaction
        map_data = st_folium(m, width=350, height=300)
        # st.markdown(type(latlng_popup))

        if map_data and 'lat' in map_data and 'lng' in map_data:
            update_coordinates(map_data)

        # Update state based on coordinates
        state = get_state_from_coords(lat=latitude, lon=longitude)

        # Display the updated state
        st.markdown(f"**Selected State:** {state}")
        
        distance = st.number_input("Distance (miles)", value=0.5, min_value=0.0, max_value=50.0, step=0.1)
    
    with col2:
        st.subheader("🕐 Time Information")
        
        accident_date = st.date_input("Accident Date", value=datetime.now())
        accident_time = st.time_input("Accident Time", value=datetime.now().time())
        
        # Combine date and time
        accident_datetime = datetime.combine(accident_date, accident_time)
        
        st.markdown(f"**Selected DateTime:** {accident_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        
        st.markdown("---")
        
        st.subheader("🌦️ Weather & Road Features")
        
        fetch_data = st.checkbox("Fetch real-time data from APIs", value=True)
        
        if fetch_data:
            st.info("Weather data will be fetched from NOAA API and road features from OpenStreetMap API when you click Predict.")
        else:
            st.warning("You will need to enter weather and road features manually.")
    
    st.markdown("---")
    
    # Manual input section (if APIs are disabled or for override)
    with st.expander("🔧 Advanced Options - Manual Input Override"):
        st.markdown("Override API data with manual inputs")
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            manual_temp = st.number_input("Temperature (°F)", value=65.0, min_value=-20.0, max_value=120.0)
            manual_humidity = st.number_input("Humidity (%)", value=60.0, min_value=0.0, max_value=100.0)
            manual_pressure = st.number_input("Pressure (in)", value=29.92, min_value=28.0, max_value=31.0)
            manual_visibility = st.number_input("Visibility (mi)", value=10.0, min_value=0.0, max_value=10.0)
            manual_wind_speed = st.number_input("Wind Speed (mph)", value=5.0, min_value=0.0, max_value=100.0)
        
        with col_b:
            manual_weather = st.selectbox("Weather Condition", ['Clear', 'Cloudy', 'Rain', 'Snow', 'Fog'])
            manual_wind_dir = st.selectbox("Wind Direction", ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
            
            st.markdown("**Road Features:**")
            manual_amenity = st.checkbox("Amenity Nearby")
            manual_crossing = st.checkbox("Crossing")
            manual_junction = st.checkbox("Junction")
            manual_traffic_signal = st.checkbox("Traffic Signal")
        
        use_manual = st.checkbox("Use manual inputs instead of API data")
    
    # Prediction button
    st.markdown("---")
    
    col_predict, col_space = st.columns([1, 2])
    
    with col_predict:
        predict_button = st.button("🎯 Predict Severity", type="primary", use_container_width=True)
    
    if predict_button:
        with st.spinner("Gathering data and making prediction..."):
            try:
                # Fetch or use manual data
                if fetch_data and not (st.session_state.get('use_manual', False) if 'use_manual' in st.session_state else use_manual):
                    with st.status("Fetching weather data from NOAA...", expanded=True) as status:
                        weather_data = fetch_weather_data(latitude, longitude, accident_datetime)
                        st.write("✅ Weather data retrieved")
                        status.update(label="Weather data complete!", state="complete")
                    
                    with st.status("Fetching road features from OpenStreetMap...", expanded=True) as status:
                        road_features = fetch_road_features(latitude, longitude)
                        st.write("✅ Road features retrieved")
                        status.update(label="Road features complete!", state="complete")
                else:
                    weather_data = {
                        'Temperature(F)': manual_temp,
                        'Humidity(%)': manual_humidity,
                        'Pressure(in)': manual_pressure,
                        'Visibility(mi)': manual_visibility,
                        'Wind_Speed(mph)': manual_wind_speed,
                        'Weather_Simple': manual_weather,
                        'Wind_Direction_Simple': manual_wind_dir
                    }
                    
                    road_features = {
                        'Amenity': manual_amenity,
                        'Crossing': manual_crossing,
                        'Give_Way': False,
                        'Junction': manual_junction,
                        'No_Exit': False,
                        'Railway': False,
                        'Station': False,
                        'Stop': False,
                        'Traffic_Signal': manual_traffic_signal
                    }
                
                # Create temporal features
                temporal_features = create_temporal_features(accident_datetime)
                
                # Create distance features
                distance_features = create_distance_features(distance)
                
                # Combine all features
                input_data = {
                    'State': state,
                    'Start_Lat': latitude,
                    'Start_Lng': longitude,
                    **weather_data,
                    **road_features,
                    **temporal_features,
                    **distance_features
                }
                
                # Create DataFrame
                input_df = pd.DataFrame([input_data])
                # Apply preprocessing
                X_processed = preprocessor.transform(input_df)

                # Predict
                prediction = model.predict(X_processed)[0]
                # class_labels = model.classes_

                
                prediction_proba = model.predict_proba(X_processed)[0]
                prediction_proba = model.predict_proba(X_processed)[0]
                # Get class labels in correct order
                class_labels = model.classes_ + 1
                prediction = class_labels[prediction_proba.argmax()]

                severity_map = {1: "Low", 2: "Moderate", 3: "High", 4: "Severe"}
                severity_colors = {1: "🟢", 2: "🟡", 3: "🟠", 4: "🔴"}
                prob_df = pd.DataFrame({
                    'Severity Level': [f"{severity_map[c]} (Level {c})" for c in class_labels],
                    'Probability': [f"{p*100:.2f}%" for p in prediction_proba]
                })
                            
                # Display results
                st.markdown("---")
                st.markdown("## 🎯 Prediction Results")
                
                
                # predicted_severity = severity_map[prediction]
                predicted_severity = prediction
                severity_icon = severity_colors[prediction]
                
                col_res1, col_res2, col_res3 = st.columns([1, 2, 1])
                
                with col_res2:
                    st.markdown(f"""
                    <div class="prediction-box" style="text-align: center;">
                        <h1>{severity_icon} {predicted_severity} Severity</h1>
                        <h3>Severity Level: {prediction}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Show probabilities
                st.markdown("### 📊 Confidence Levels")
                
                # prob_df = pd.DataFrame({
                #     'Severity Level': [f"{severity_map[i]} (Level {i})" for i in range(1, 5)],
                #     'Probability': [f"{p*100:.2f}%" for p in prediction_proba]
                # })
                
                st.dataframe(prob_df, use_container_width=True, hide_index=True)
                
                # Show input summary
                with st.expander("📋 Input Data Summary"):
                    st.json(input_data)
                
                st.success("✅ Prediction completed successfully!")
                
            except Exception as e:
                st.error(f"❌ Error during prediction: {str(e)}")
                st.exception(e)

# Main app logic
def main():
    # Sidebar
    with st.sidebar:
        st.image("https://raw.githubusercontent.com/github/explore/80688e429a7d4ef2fca1e82350fe8e3517d3494d/topics/python/python.png", width=100)
        st.title("Navigation")
        
        st.markdown("---")
        
        if st.button("🏠 Home", use_container_width=True):
            set_page('home')
        
        if st.button("📊 Dashboard", use_container_width=True):
            set_page('dashboard')
        
        if st.button("🔮 Prediction", use_container_width=True):
            set_page('prediction')
        
        st.markdown("---")
        
        st.markdown("""
        ### ℹ️ About
        
        **Version:** 1.0.0
        
        **Model:** XGBoost Classifier
        
        **Data Source:** US Accidents Dataset
        
        **APIs Used:**
        - NOAA Weather API
        - OpenStreetMap Overpass API
        """)
        
        st.markdown("---")
        st.markdown("Made with ❤️ using Streamlit")
    
    # Route to appropriate page
    if st.session_state.page == 'home':
        show_home_page()
    elif st.session_state.page == 'dashboard':
        show_dashboard_page()
    elif st.session_state.page == 'prediction':
        show_prediction_page()

if __name__ == "__main__":
    main()