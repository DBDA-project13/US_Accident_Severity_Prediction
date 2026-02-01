import numpy as np # type: ignore
import pandas as pd # type: ignore

# Feature Engineering
def simplify_weather(cond: str) -> str:
    """
    Map raw weather description to simplified categories.
    """
    cond = str(cond).lower()

    if "snow" in cond or "blowing snow" in cond or "drifting snow" in cond:
        return "snow"
    if "rain" in cond or "drizzle" in cond or "shower" in cond:
        return "rain"
    if "storm" in cond or "thunder" in cond or "t-storm" in cond:
        return "storm"
    if "fog" in cond or "mist" in cond or "haze" in cond:
        return "fog"
    if "cloud" in cond or "overcast" in cond:
        return "cloudy"
    if "clear" in cond or "fair" in cond:
        return "clear"
    if "hail" in cond or "ice pellet" in cond:
        return "hail"
    if "sleet" in cond or "freezing rain" in cond or "freezing drizzle" in cond:
        return "sleet"
    if "dust" in cond or "sand" in cond or "ash" in cond:
        return "dust/sand"
    if "smoke" in cond:
        return "smoke"
    if "tornado" in cond or "funnel" in cond:
        return "tornado"
    return "other"

def hour_to_time_bucket(hour):
    """
    Map hour (0–23) to a coarse time-of-day bucket.
    Used later for aggregation.
    """
    if 0 <= hour < 4:
        return "Midnight"
    elif 4 <= hour < 11:
        return "Morning"
    elif 11 <= hour < 15:
        return "Afternoon"
    elif 15 <= hour < 18:
        return "Evening"
    else:
        return "Night"


def transform_distance(df, col='Distance(mi)'):
    """
    Transform the Distance(mi) column:
    1. Cap extreme outliers at 5 miles
    2. Log-transform to reduce skewness
    3. Bin into categories for interpretability
    Returns the dataframe with new features.
    """
    # Cap outliers
    df[f'{col}_capped'] = np.where(df[col] > 5, 5, df[col])
    
    # Log transform
    df[f'{col}_log'] = np.log1p(df[f'{col}_capped'])
    
    # Bin into categories
    bins = [0, 0.01, 0.1, 1, 5]
    labels = ['Zero', 'Very_Short', 'Short', 'Medium']
    df[f'{col}_bin'] = pd.cut(df[f'{col}_capped'], bins=bins, labels=labels, include_lowest=True)
    
    return df


def wind_direction_mapping(direction: str) -> str:
    """
    Map wind direction to a standardized set of values.
    """
    direction = str(direction).upper()
    if direction in ['N', 'NNE', 'NNW', 'NORTH']:
        return 'N'
    elif direction in ['S', 'SSE', 'SSW', 'SOUTH']:
        return 'S'
    elif direction in ['E', 'ENE', 'ESE', 'EAST']:
        return 'E'
    elif direction in ['W', 'WNW', 'WSW', 'WEST']:
        return 'W'
    elif direction in ['NE', 'SE', 'SW', 'NW']:
        return direction
    elif direction in ['CALM']:
        return 'CALM'
    elif direction in ['VARIABLE', 'VAR']:
        return 'VAR'
    else:
        return 'MISSING'
    

def month_to_season(month):
    """
    month to Season
    """
    if month in [12, 1, 2]:
        return "winter"
    elif month in [3, 4, 5]:
        return "spring"
    elif month in [6, 7, 8]:
        return "summer"
    elif month in [9, 10, 11]:
        return "fall"
    else:
        return month
    
def is_rushhour(hour):
    if hour in [7, 8, 9, 16, 17, 18, 19]:
        return True
    else:
        return False