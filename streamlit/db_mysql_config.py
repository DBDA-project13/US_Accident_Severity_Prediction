"""
AWS RDS MySQL Database Configuration
"""
import mysql.connector
from mysql.connector import Error
import os
from datetime import datetime
import json
import streamlit as st

class AccidentPredictionDB:
    """MySQL Database handler for accident prediction storage"""
    
    def __init__(self):
        """Initialize database connection"""
        self.connection = None
        self.cursor = None
        
    def connect(self):
        """Establish connection to AWS RDS MySQL"""
        try:
            self.connection = mysql.connector.connect(
                host=os.getenv('RDS_HOSTNAME', 'accident-prediction-db.cdgemicqmj0e.ap-south-1.rds.amazonaws.com'),
                port=os.getenv('RDS_PORT', '3306'),
                database=os.getenv('RDS_DB_NAME', 'accident_predictions'),
                user=os.getenv('RDS_USERNAME', 'admin'),
                password=os.getenv('RDS_PASSWORD', 'dbda8414952')
            )
            
            if self.connection.is_connected():
                self.cursor = self.connection.cursor(dictionary=True)
                print("Successfully connected to MySQL database")
                return True
        except Error as e:
            st.error(f"Database connection failed: {e}")
            return False
    
    def create_tables(self):
        """Create necessary tables if they don't exist"""
        create_table_query = """
        CREATE TABLE IF NOT EXISTS accident_predictions (
            id INT AUTO_INCREMENT PRIMARY KEY,
            prediction_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            
            -- Location Information
            latitude DECIMAL(10, 7) NOT NULL,
            longitude DECIMAL(10, 7) NOT NULL,
            state VARCHAR(2),
            distance_mi DECIMAL(10, 2),
            
            -- Temporal Information
            accident_datetime DATETIME NOT NULL,
            hour INT,
            day INT,
            month INT,
            dayofweek INT,
            season VARCHAR(20),
            is_rushhour BOOLEAN,
            is_holiday BOOLEAN,
            time_bucket VARCHAR(20),
            
            -- Weather Information
            temperature_f DECIMAL(5, 2),
            humidity_pct DECIMAL(5, 2),
            pressure_in DECIMAL(5, 2),
            visibility_mi DECIMAL(5, 2),
            wind_speed_mph DECIMAL(5, 2),
            weather_condition VARCHAR(50),
            wind_direction VARCHAR(10),
            
            -- Road Features
            amenity BOOLEAN DEFAULT FALSE,
            crossing BOOLEAN DEFAULT FALSE,
            give_way BOOLEAN DEFAULT FALSE,
            junction BOOLEAN DEFAULT FALSE,
            no_exit BOOLEAN DEFAULT FALSE,
            railway BOOLEAN DEFAULT FALSE,
            station BOOLEAN DEFAULT FALSE,
            stop_sign BOOLEAN DEFAULT FALSE,
            traffic_signal BOOLEAN DEFAULT FALSE,
            
            -- Prediction Results
            predicted_severity INT NOT NULL,
            severity_label VARCHAR(20),
            confidence_level_1 DECIMAL(5, 4),
            confidence_level_2 DECIMAL(5, 4),
            confidence_level_3 DECIMAL(5, 4),
            confidence_level_4 DECIMAL(5, 4),
            
            -- Metadata
            data_source VARCHAR(20),
            user_session_id VARCHAR(100),
            
            -- Timestamps
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            
            INDEX idx_prediction_timestamp (prediction_timestamp),
            INDEX idx_state (state),
            INDEX idx_severity (predicted_severity),
            INDEX idx_accident_datetime (accident_datetime)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """
        
        try:
            self.cursor.execute(create_table_query)
            self.connection.commit()
            print("Table created successfully")
            return True
        except Error as e:
            st.error(f"Error creating tables: {e}")
            return False
    
    def insert_prediction(self, input_data, prediction, prediction_proba, data_source='API'):
        """
        Insert prediction record into database
        
        Args:
            input_data: Dictionary containing all input features
            prediction: Predicted severity level (1-4)
            prediction_proba: Array of probabilities for each class
            data_source: 'API' or 'Manual'
        
        Returns:
            int: ID of inserted record, or None if failed
        """
        insert_query = """
        INSERT INTO accident_predictions (
            latitude, longitude, state, distance_mi,
            accident_datetime, hour, day, month, dayofweek, season,
            is_rushhour, is_holiday, time_bucket,
            temperature_f, humidity_pct, pressure_in, visibility_mi,
            wind_speed_mph, weather_condition, wind_direction,
            amenity, crossing, give_way, junction, no_exit,
            railway, station, stop_sign, traffic_signal,
            predicted_severity, severity_label,
            confidence_level_1, confidence_level_2, confidence_level_3, confidence_level_4,
            data_source, user_session_id
        ) VALUES (
            %(Start_Lat)s, %(Start_Lng)s, %(State)s, %(Distance_mi)s,
            %(accident_datetime)s, %(hour)s, %(day)s, %(month)s, %(dayofweek)s, %(season)s,
            %(is_rushhour)s, %(is_holiday)s, %(time_bucket)s,
            %(temperature)s, %(humidity)s, %(pressure)s, %(visibility)s,
            %(wind_speed)s, %(weather)s, %(wind_direction)s,
            %(Amenity)s, %(Crossing)s, %(Give_Way)s, %(Junction)s, %(No_Exit)s,
            %(Railway)s, %(Station)s, %(Stop)s, %(Traffic_Signal)s,
            %(predicted_severity)s, %(severity_label)s,
            %(conf_1)s, %(conf_2)s, %(conf_3)s, %(conf_4)s,
            %(data_source)s, %(session_id)s
        )
        """
        
        # Map severity to label
        severity_map = {1: "Low", 2: "Moderate", 3: "High", 4: "Severe"}
        
        # Prepare data
        data = {
            'Start_Lat': input_data.get('Start_Lat'),
            'Start_Lng': input_data.get('Start_Lng'),
            'State': input_data.get('State'),
            'Distance_mi': input_data.get('Distance(mi)_capped', 0),
            'accident_datetime': input_data.get('accident_datetime'),
            'hour': input_data.get('hour'),
            'day': input_data.get('day'),
            'month': input_data.get('month'),
            'dayofweek': input_data.get('dayofweek'),
            'season': input_data.get('season'),
            'is_rushhour': input_data.get('is_rushhour', False),
            'is_holiday': input_data.get('is_holiday', False),
            'time_bucket': input_data.get('time_bucket'),
            'temperature': input_data.get('Temperature(F)'),
            'humidity': input_data.get('Humidity(%)'),
            'pressure': input_data.get('Pressure(in)'),
            'visibility': input_data.get('Visibility(mi)'),
            'wind_speed': input_data.get('Wind_Speed(mph)'),
            'weather': input_data.get('Weather_Simple'),
            'wind_direction': input_data.get('Wind_Direction_Simple'),
            'Amenity': input_data.get('Amenity', False),
            'Crossing': input_data.get('Crossing', False),
            'Give_Way': input_data.get('Give_Way', False),
            'Junction': input_data.get('Junction', False),
            'No_Exit': input_data.get('No_Exit', False),
            'Railway': input_data.get('Railway', False),
            'Station': input_data.get('Station', False),
            'Stop': input_data.get('Stop', False),
            'Traffic_Signal': input_data.get('Traffic_Signal', False),
            'predicted_severity': int(prediction),
            'severity_label': severity_map.get(prediction, 'Unknown'),
            'conf_1': float(prediction_proba[0]) if len(prediction_proba) > 0 else 0,
            'conf_2': float(prediction_proba[1]) if len(prediction_proba) > 1 else 0,
            'conf_3': float(prediction_proba[2]) if len(prediction_proba) > 2 else 0,
            'conf_4': float(prediction_proba[3]) if len(prediction_proba) > 3 else 0,
            'data_source': data_source,
            'session_id': st.session_state.get('session_id', 'unknown')
        }
        
        try:
            self.cursor.execute(insert_query, data)
            self.connection.commit()
            record_id = self.cursor.lastrowid
            return record_id
        except Error as e:
            st.error(f"Error inserting prediction: {e}")
            self.connection.rollback()
            return None
    
    def get_recent_predictions(self, limit=10):
        """Get recent predictions"""
        query = """
        SELECT 
            id, prediction_timestamp, latitude, longitude, state,
            accident_datetime, predicted_severity, severity_label,
            confidence_level_1, confidence_level_2, confidence_level_3, confidence_level_4
        FROM accident_predictions
        ORDER BY prediction_timestamp DESC
        LIMIT %s
        """
        
        try:
            self.cursor.execute(query, (limit,))
            return self.cursor.fetchall()
        except Error as e:
            st.error(f"Error fetching predictions: {e}")
            return []
    
    def get_statistics(self):
        """Get prediction statistics"""
        query = """
        SELECT 
            COUNT(*) as total_predictions,
            AVG(predicted_severity) as avg_severity,
            SUM(CASE WHEN predicted_severity = 1 THEN 1 ELSE 0 END) as severity_1_count,
            SUM(CASE WHEN predicted_severity = 2 THEN 1 ELSE 0 END) as severity_2_count,
            SUM(CASE WHEN predicted_severity = 3 THEN 1 ELSE 0 END) as severity_3_count,
            SUM(CASE WHEN predicted_severity = 4 THEN 1 ELSE 0 END) as severity_4_count
        FROM accident_predictions
        """
        
        try:
            self.cursor.execute(query)
            return self.cursor.fetchone()
        except Error as e:
            st.error(f"Error fetching statistics: {e}")
            return None
    
    def close(self):
        """Close database connection"""
        if self.cursor:
            self.cursor.close()
        if self.connection and self.connection.is_connected():
            self.connection.close()
            print("MySQL connection closed")
    
    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


# Initialize database session
def init_db_session():
    """Initialize database session in Streamlit"""
    if 'session_id' not in st.session_state:
        import uuid
        st.session_state['session_id'] = str(uuid.uuid4())
    
    if 'db' not in st.session_state:
        st.session_state['db'] = AccidentPredictionDB()
        if st.session_state['db'].connect():
            st.session_state['db'].create_tables()