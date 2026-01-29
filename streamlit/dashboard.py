"""
Dashboard module for US Accident Severity Prediction System

This is a sample dashboard. Replace with your actual dashboard implementation.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

def show_dashboard():
    """Main dashboard display function"""
    
    st.markdown("## 📊 Accident Analysis Dashboard")
    
    # Sample data - replace with actual data loading
    @st.cache_data
    def load_dashboard_data():
        # This is sample data - replace with your actual data loading
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        sample_data = pd.DataFrame({
            'Date': dates,
            'Accidents': np.random.randint(50, 200, size=len(dates)),
            'Severity_1': np.random.randint(20, 100, size=len(dates)),
            'Severity_2': np.random.randint(15, 60, size=len(dates)),
            'Severity_3': np.random.randint(10, 30, size=len(dates)),
            'Severity_4': np.random.randint(5, 15, size=len(dates)),
        })
        return sample_data
    
    # Load data
    df = load_dashboard_data()
    
    # Key Metrics
    st.markdown("### 📈 Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_accidents = df['Accidents'].sum()
        st.metric(
            label="Total Accidents",
            value=f"{total_accidents:,}",
            delta=f"+{int(total_accidents * 0.05):,} from last period"
        )
    
    with col2:
        avg_severity = (
            df['Severity_1'].sum() * 1 +
            df['Severity_2'].sum() * 2 +
            df['Severity_3'].sum() * 3 +
            df['Severity_4'].sum() * 4
        ) / df['Accidents'].sum()
        st.metric(
            label="Avg Severity",
            value=f"{avg_severity:.2f}",
            delta="-0.15",
            delta_color="inverse"
        )
    
    with col3:
        high_severity = df['Severity_4'].sum()
        st.metric(
            label="Severe Accidents",
            value=f"{high_severity:,}",
            delta=f"+{int(high_severity * 0.12):,}"
        )
    
    with col4:
        low_severity = df['Severity_1'].sum()
        st.metric(
            label="Minor Accidents",
            value=f"{low_severity:,}",
            delta=f"-{int(low_severity * 0.08):,}",
            delta_color="inverse"
        )
    
    st.markdown("---")
    
    # Time series analysis
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📅 Accidents Over Time")
        
        fig_timeline = go.Figure()
        
        fig_timeline.add_trace(go.Scatter(
            x=df['Date'],
            y=df['Accidents'],
            mode='lines',
            name='Total Accidents',
            line=dict(color='#1f77b4', width=2),
            fill='tonexty'
        ))
        
        fig_timeline.update_layout(
            height=400,
            hovermode='x unified',
            xaxis_title='Date',
            yaxis_title='Number of Accidents'
        )
        
        st.plotly_chart(fig_timeline, use_container_width=True)
    
    with col2:
        st.markdown("### 🎯 Severity Distribution")
        
        severity_data = pd.DataFrame({
            'Severity': ['Low (1)', 'Moderate (2)', 'High (3)', 'Severe (4)'],
            'Count': [
                df['Severity_1'].sum(),
                df['Severity_2'].sum(),
                df['Severity_3'].sum(),
                df['Severity_4'].sum()
            ]
        })
        
        colors_map = {
            'Low (1)': '#28a745',
            'Moderate (2)': '#ffc107',
            'High (3)': '#fd7e14',
            'Severe (4)': '#dc3545'
        }
        
        fig_pie = px.pie(
            severity_data,
            values='Count',
            names='Severity',
            color='Severity',
            color_discrete_map=colors_map,
            hole=0.4
        )
        
        fig_pie.update_layout(height=400)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    st.markdown("---")
    
    # Hourly and daily patterns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ⏰ Accidents by Hour of Day")
        
        # Sample hourly data
        hourly_data = pd.DataFrame({
            'Hour': range(24),
            'Accidents': [30, 25, 20, 18, 22, 35, 65, 95, 85, 70, 75, 80,
                         85, 80, 75, 85, 110, 125, 105, 85, 70, 55, 45, 35]
        })
        
        fig_hourly = px.bar(
            hourly_data,
            x='Hour',
            y='Accidents',
            color='Accidents',
            color_continuous_scale='Reds'
        )
        
        fig_hourly.update_layout(height=350)
        st.plotly_chart(fig_hourly, use_container_width=True)
    
    with col2:
        st.markdown("### 📆 Accidents by Day of Week")
        
        # Sample daily data
        daily_data = pd.DataFrame({
            'Day': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
            'Accidents': [850, 920, 880, 910, 1100, 750, 650]
        })
        
        fig_daily = px.bar(
            daily_data,
            x='Day',
            y='Accidents',
            color='Accidents',
            color_continuous_scale='Blues'
        )
        
        fig_daily.update_layout(height=350)
        st.plotly_chart(fig_daily, use_container_width=True)
    
    st.markdown("---")
    
    # State-wise analysis
    st.markdown("### 🗺️ State-wise Accident Statistics")
    
    # Sample state data
    state_data = pd.DataFrame({
        'State': ['CA', 'TX', 'FL', 'NY', 'PA', 'IL', 'OH', 'GA', 'NC', 'MI'],
        'Accidents': [15000, 12500, 11000, 9500, 8500, 7800, 7200, 6800, 6200, 5900],
        'Avg_Severity': [2.3, 2.1, 2.4, 2.2, 2.0, 2.3, 2.1, 2.2, 2.0, 2.1]
    })
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        fig_state = px.bar(
            state_data,
            x='State',
            y='Accidents',
            title='Top 10 States by Accident Count',
            color='Accidents',
            color_continuous_scale='Viridis'
        )
        fig_state.update_layout(height=400)
        st.plotly_chart(fig_state, use_container_width=True)
    
    with col2:
        fig_severity_state = px.scatter(
            state_data,
            x='Accidents',
            y='Avg_Severity',
            size='Accidents',
            color='Avg_Severity',
            hover_data=['State'],
            title='Accidents vs Average Severity by State',
            color_continuous_scale='RdYlGn_r'
        )
        fig_severity_state.update_layout(height=400)
        st.plotly_chart(fig_severity_state, use_container_width=True)
    
    st.markdown("---")
    
    # Weather impact
    st.markdown("### 🌤️ Weather Impact on Accidents")
    
    weather_data = pd.DataFrame({
        'Weather': ['Clear', 'Cloudy', 'Rain', 'Snow', 'Fog'],
        'Accidents': [45000, 28000, 18000, 5000, 4000],
        'Avg_Severity': [1.9, 2.1, 2.5, 2.8, 2.9]
    })
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_weather = px.bar(
            weather_data,
            x='Weather',
            y='Accidents',
            title='Accidents by Weather Condition',
            color='Avg_Severity',
            color_continuous_scale='Reds'
        )
        fig_weather.update_layout(height=350)
        st.plotly_chart(fig_weather, use_container_width=True)
    
    with col2:
        fig_weather_severity = px.scatter(
            weather_data,
            x='Accidents',
            y='Avg_Severity',
            size='Accidents',
            color='Weather',
            title='Weather Condition Impact',
            text='Weather'
        )
        fig_weather_severity.update_traces(textposition='top center')
        fig_weather_severity.update_layout(height=350)
        st.plotly_chart(fig_weather_severity, use_container_width=True)
    
    # Additional insights
    st.markdown("---")
    st.markdown("### 💡 Key Insights")
    
    insights_col1, insights_col2, insights_col3 = st.columns(3)
    
    with insights_col1:
        st.info("""
        **Peak Hours**
        
        Rush hours (5-7 PM) show 40% higher accident rates compared to off-peak hours.
        """)
    
    with insights_col2:
        st.warning("""
        **Weather Impact**
        
        Accidents in adverse weather conditions are 50% more likely to be severe.
        """)
    
    with insights_col3:
        st.success("""
        **Weekend Trend**
        
        Weekends see 30% fewer accidents but with slightly higher severity rates.
        """)
    
    # Data table
    with st.expander("📋 View Raw Data Sample"):
        st.dataframe(df.head(20), use_container_width=True)

if __name__ == "__main__":
    import numpy as np
    show_dashboard()
