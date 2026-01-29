"""
Utility functions for the US Accident Severity Prediction System
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime
import streamlit as st

class WeatherAPI:
    """Handler for NOAA Weather API calls"""
    
    def __init__(self, api_token):
        self.api_token = api_token
        self.base_url = "https://www.ncdc.noaa.gov/cdo-web/api/v2/data"
    
    def fetch_weather(self, lat, lon, date_time):
        """
        Fetch weather data from NOAA API
        
        Args:
            lat (float): Latitude
            lon (float): Longitude
            date_time (datetime): Date and time of accident
            
        Returns:
            dict: Weather data or default values
        """
        try:
            headers = {'token': self.api_token}
            date_str = date_time.strftime('%Y-%m-%d')
            
            params = {
                'datasetid': 'GHCND',
                'locationid': 'FIPS:US',
                'startdate': date_str,
                'enddate': date_str,
                'limit': 1000,
                'units': 'standard'
            }
            
            response = requests.get(
                self.base_url,
                headers=headers,
                params=params,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                return self._parse_weather_response(data)
            else:
                return self._get_default_weather()
                
        except Exception as e:
            st.warning(f"Weather API error: {e}")
            return self._get_default_weather()
    
    def _parse_weather_response(self, data):
        """Parse NOAA API response into required format"""
        # This is a simplified parser - enhance based on actual NOAA response structure
        weather_data = {
            'Temperature(F)': 65.0,
            'Humidity(%)': 60.0,
            'Pressure(in)': 29.92,
            'Visibility(mi)': 10.0,
            'Wind_Speed(mph)': 5.0,
            'Weather_Simple': 'Clear',
            'Wind_Direction_Simple': 'N'
        }
        
        # Extract actual values from API response if available
        results = data.get('results', [])
        for result in results:
            datatype = result.get('datatype', '')
            value = result.get('value', 0)
            
            if 'TEMP' in datatype or 'TMAX' in datatype:
                # Convert from Celsius to Fahrenheit if needed
                weather_data['Temperature(F)'] = value
            elif 'AWND' in datatype:  # Average wind speed
                weather_data['Wind_Speed(mph)'] = value
            # Add more parsing logic as needed
        
        return weather_data
    
    def _get_default_weather(self):
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


class RoadFeaturesAPI:
    """Handler for OpenStreetMap Overpass API calls"""
    
    def __init__(self, overpass_url="http://overpass-api.de/api/interpreter", radius=100):
        self.overpass_url = overpass_url
        self.radius = radius
    
    def fetch_road_features(self, lat, lon):
        """
        Fetch road features from OpenStreetMap
        
        Args:
            lat (float): Latitude
            lon (float): Longitude
            
        Returns:
            dict: Road features (boolean flags)
        """
        try:
            query = self._build_overpass_query(lat, lon)
            response = requests.post(
                self.overpass_url,
                data={'data': query},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                return self._parse_osm_response(data)
            else:
                return self._get_default_features()
                
        except Exception as e:
            st.warning(f"OpenStreetMap API error: {e}")
            return self._get_default_features()
    
    def _build_overpass_query(self, lat, lon):
        """Build Overpass QL query"""
        query = f"""
        [out:json];
        (
          node["amenity"](around:{self.radius},{lat},{lon});
          node["highway"="crossing"](around:{self.radius},{lat},{lon});
          node["highway"="give_way"](around:{self.radius},{lat},{lon});
          node["highway"="stop"](around:{self.radius},{lat},{lon});
          node["highway"="traffic_signals"](around:{self.radius},{lat},{lon});
          node["railway"](around:{self.radius},{lat},{lon});
          node["railway"="station"](around:{self.radius},{lat},{lon});
          way["junction"](around:{self.radius},{lat},{lon});
          way["noexit"="yes"](around:{self.radius},{lat},{lon});
        );
        out body;
        """
        return query
    
    def _parse_osm_response(self, data):
        """Parse OpenStreetMap response"""
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
            if tags.get('noexit') == 'yes':
                features['No_Exit'] = True
        
        return features
    
    def _get_default_features(self):
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


class FeatureEngineering:
    """Feature engineering utilities"""
    
    @staticmethod
    def create_temporal_features(date_time):
        """
        Create temporal features from datetime
        
        Args:
            date_time (datetime): Accident datetime
            
        Returns:
            dict: Temporal features
        """
        features = {
            'hour': date_time.hour,
            'day': date_time.day,
            'month': date_time.month,
            'dayofweek': date_time.weekday(),
            'season': FeatureEngineering.get_season(date_time.month)
        }
        
        # Rush hour indicator (7-9 AM, 4-7 PM)
        features['is_rushhour'] = (
            (7 <= date_time.hour <= 9) or 
            (16 <= date_time.hour <= 19)
        )
        
        # Holiday indicator (simplified - enhance with holiday calendar)
        features['is_holiday'] = False
        
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
    
    @staticmethod
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
    
    @staticmethod
    def create_distance_features(distance):
        """
        Create distance-related features
        
        Args:
            distance (float): Distance in miles
            
        Returns:
            dict: Distance features
        """
        features = {
            'Distance(mi)_capped': min(distance, 10.0),
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
    
    @staticmethod
    def prepare_input_dataframe(location_data, weather_data, road_features, 
                                temporal_features, distance_features):
        """
        Combine all features into a single DataFrame
        
        Args:
            location_data (dict): State, Lat, Lng
            weather_data (dict): Weather features
            road_features (dict): Road boolean features
            temporal_features (dict): Time-based features
            distance_features (dict): Distance features
            
        Returns:
            pd.DataFrame: Complete input data
        """
        input_data = {
            **location_data,
            **weather_data,
            **road_features,
            **temporal_features,
            **distance_features
        }
        
        return pd.DataFrame([input_data])


class PredictionVisualizer:
    """Visualization utilities for predictions"""
    
    SEVERITY_CONFIG = {
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
    
    @staticmethod
    def get_severity_info(severity_level):
        """Get information about a severity level"""
        return PredictionVisualizer.SEVERITY_CONFIG.get(severity_level, {})
    
    @staticmethod
    def format_prediction_box(severity_level):
        """Create HTML for prediction result box"""
        info = PredictionVisualizer.get_severity_info(severity_level)
        
        return f"""
        <div style="
            padding: 30px;
            border-radius: 15px;
            background: linear-gradient(135deg, {info['color']}22 0%, {info['color']}44 100%);
            border: 3px solid {info['color']};
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        ">
            <h1 style="margin: 0; font-size: 3rem;">{info['icon']}</h1>
            <h2 style="margin: 10px 0; color: {info['color']};">{info['name']} Severity</h2>
            <h3 style="margin: 5px 0; color: #666;">Level {severity_level}</h3>
            <p style="margin: 15px 0; color: #555; font-size: 1.1rem;">{info['description']}</p>
        </div>
        """
    
    @staticmethod
    def create_probability_chart(probabilities):
        """Create a visual chart of prediction probabilities"""
        import plotly.graph_objects as go
        
        severity_names = [
            f"{PredictionVisualizer.SEVERITY_CONFIG[i]['name']} ({i})" 
            for i in range(1, 5)
        ]
        colors = [
            PredictionVisualizer.SEVERITY_CONFIG[i]['color'] 
            for i in range(1, 5)
        ]
        
        fig = go.Figure(data=[
            go.Bar(
                x=severity_names,
                y=probabilities * 100,
                text=[f'{p:.1f}%' for p in probabilities * 100],
                textposition='auto',
                marker_color=colors
            )
        ])
        
        fig.update_layout(
            title='Prediction Confidence by Severity Level',
            xaxis_title='Severity Level',
            yaxis_title='Probability (%)',
            yaxis=dict(range=[0, 100]),
            height=400
        )
        
        return fig


# Validation utilities
def validate_coordinates(lat, lon):
    """Validate latitude and longitude"""
    if not (-90 <= lat <= 90):
        return False, "Latitude must be between -90 and 90"
    if not (-180 <= lon <= 180):
        return False, "Longitude must be between -180 and 180"
    return True, "Valid coordinates"


def validate_us_coordinates(lat, lon):
    """Validate coordinates are within continental US bounds"""
    if not (24.0 <= lat <= 50.0):
        return False, "Latitude must be within continental US (24°N - 50°N)"
    if not (-125.0 <= lon <= -66.0):
        return False, "Longitude must be within continental US (125°W - 66°W)"
    return True, "Valid US coordinates"
