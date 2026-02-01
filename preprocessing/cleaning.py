import pandas as pd
import numpy as np # type: ignore
import holidays# type: ignore

from feature import *
# from feature import simplify_weather, hour_to_time_bucket, transform_distance, wind_direction_mapping

print("loading dataset")
df = pd.read_csv("data/raw/US_Accidents_March23.csv")
print("loaded dataset", df.shape)

# Compact memory: downcast numeric columns where safe
for col in df.select_dtypes(include=['int64']).columns:
    df[col] = pd.to_numeric(df[col], downcast='integer')

for col in df.select_dtypes(include=['float64']).columns:
    df[col] = pd.to_numeric(df[col], downcast='float')
print("downcasted loaded dataset")
# EDA
####print(df.columns)
all_columns = ['ID', 'Source', 'Severity', 'Start_Time', 'End_Time', 'Start_Lat',
'Start_Lng', 'End_Lat', 'End_Lng', 'Distance(mi)', 'Description',
'Street', 'City', 'County', 'State', 'Zipcode', 'Country', 'Timezone',
'Airport_Code', 'Weather_Timestamp', 'Temperature(F)', 'Wind_Chill(F)',
'Humidity(%)', 'Pressure(in)', 'Visibility(mi)', 'Wind_Direction',
'Wind_Speed(mph)', 'Precipitation(in)', 'Weather_Condition', 'Amenity',
'Bump', 'Crossing', 'Give_Way', 'Junction', 'No_Exit', 'Railway',
'Roundabout', 'Station', 'Stop', 'Traffic_Calming', 'Traffic_Signal',
'Turning_Loop', 'Sunrise_Sunset', 'Civil_Twilight', 'Nautical_Twilight',
'Astronomical_Twilight']

numerical_cols = ['Distance(mi)', 'Temperature(F)', 
                  'Humidity(%)', 'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)']
categorical_cols = ['Severity', 'Source', 'City', 'County', 'State', 'Country', 'Timezone', 'Airport_Code', 'Wind_Direction', 
                    'Weather_Condition', 'Sunrise_Sunset', 'Civil_Twilight', 'Nautical_Twilight', 'Astronomical_Twilight']
object_cols = ['ID', 'Start_Time', 'End_Time', 'Description', 'Street', 'Zipcode', 'Weather_Timestamp']
bool_cols = ['Amenity','Bump','Crossing','Give_Way','Junction','No_Exit','Railway','Roundabout','Station',
             'Stop','Traffic_Calming','Traffic_Signal','Turning_Loop'] 

# for column in df.columns:
    # if df[column].dtype == 'object':
        ####print(column)
        ###print(df[column].unique())
        ###print()

###print("Columns with a single unique value:")
# for column in df.columns:
    # if len(df[column].unique()) == 1:
        ###print(column)

###print(df)

###print(df.info())
###print()
###print(df.isna().sum())
###print(df.isna().mean())

# for key, value in df.isna().mean().items():
    # if value > 0.2:
        ###print(f"Column: {key}, Missing Percentage: {value * 100:.2f}%")
not_used_columns = ['ID', 'Source', 'End_Time', 'End_Lat', 'End_Lng',
'Description', 'Street', 'Country', 'Airport_Code', 'Weather_Timestamp',
'County', 'Zipcode', 'Timezone', 'City','Turning_Loop', 'Traffic_Calming', 'Roundabout', 'Bump']

bool_columns = ['Sunrise_Sunset', 'Civil_Twilight', 'Nautical_Twilight',
'Astronomical_Twilight'] 

high_null_columns = ['Wind_Chill(F)', 'Precipitation(in)']

df = df.drop(columns=not_used_columns)
df = df.drop(columns=high_null_columns)
df = df.drop(columns=bool_columns)
# Drop rows where location/time info is missing
df = df.dropna(subset=["State", "Start_Time"])

print()
print("dropped columns")
print()
print(df.info())

###print("After dropping NA in [State, Start_Time]:", df.shape)

# pd.set_option("display.max_columns", None)
# ###print(df.head(25))
# ###print(df.isna().sum())


# missing_value_columns = []
# for key, value in df.isna().mean().items():
    # if value > 0:
        ###print(f"Column: {key}, Missing Percentage: {value * 100:.2f}%")
        # missing_value_columns.append(key)



# ###print(df[['Start_Time', 'Sunrise_Sunset', 'Civil_Twilight', 'Nautical_Twilight', 'Astronomical_Twilight']].head(25))

# outlier detection
def detect_outliers_iqr(data, k=1.5):
    """
    Detect outliers in numerical columns using IQR.
    Returns a dictionary with column names and outlier indices.
    """
    outliers = {}
    # numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in data.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - k * IQR
        upper_bound = Q3 + k * IQR
        outlier_idx = data[(data[col] < lower_bound) | (data[col] > upper_bound)].index
        outliers[col] = outlier_idx.tolist()
    
    return outliers

# outliers imputation
def cap_outliers_iqr(series, k=1.5):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - k * IQR
    upper_bound = Q3 + k * IQR
    return np.clip(series, lower_bound, upper_bound)


print()
print("outliers")
outliers = detect_outliers_iqr(df[numerical_cols])
for key, value in outliers.items():
  print(key, len(value))
print()
  # print(outliers)


# filling Missing Values

# for weather condition
df["Weather_Condition"] = df["Weather_Condition"].fillna("Unknown")

# filling with mean, mode
dict1 = {                 
    "Temperature(F)": "median",
    "Humidity(%)": "median",
    "Pressure(in)": "median",
    "Visibility(mi)": "median",
    "Wind_Direction" : "mode",
    "Wind_Speed(mph)": "median",
}

# Apply the filling strategy
print()
print("null values")
print()
# print(df.info())
print(df.isna().sum())
# Step 0: Fill missing values
for col, strategy in dict1.items():
    if col in df.columns:
        if strategy == "median":
            df[col] = df[col].fillna(df[col].median())
        elif strategy == "mean":
            df[col] = df[col].fillna(df[col].mean())
        elif strategy == "mode":
            df[col] = df[col].fillna(df[col].mode().iloc[0])
print()
print("filled null values")
# print(df.info())
print(df.isna().sum())
print()


# Step 1: Cap outliers
for col in numerical_cols:
    df[col] = cap_outliers_iqr(df[col])

print()
print("outliers capped")
outliers = detect_outliers_iqr(df[numerical_cols])
for key, value in outliers.items():
  print(key, len(value))
print()

df = transform_distance(df)
# Step 2: Apply log transform (add +1 to avoid log(0))
# df['Distance(mi)_log'] = np.log1p(df['Distance(mi)'])
for col in numerical_cols:
    df[col] = np.log1p(df[col])

print("dropping null again")
print(df.isna().sum())
df = df.dropna()
df = df.reset_index(drop=True)
print()
print("after dropping null again")
print(df.isna().sum())
###print("After filling missing values:")
###print(df.isna().sum())
###print(df.isna().mean())

# Removing ['Sunrise_Sunset', 'Civil_Twilight', 'Nautical_Twilight', 'Astronomical_Twilight'] 
# since we are capturing it using hour_to_time_bucket as time-of-day buckets


# The distribution is heavily skewed (most accidents have near-zero distance).

df["Weather_Simple"] = df["Weather_Condition"].apply(simplify_weather)

df["Wind_Direction_Simple"] = df["Wind_Direction"].apply(wind_direction_mapping)

# Convert Start_Time to datetime; remove rows that fail to parse
df["Start_Time"] = df["Start_Time"].str.slice(0, 19)
df["Start_Time"] = pd.to_datetime(df["Start_Time"], errors="coerce")
# df = df.dropna(subset=["Start_Time"])
###print("After parsing Start_Time to datetime:", df.shape)

# Extract date and hour (used for panel aggregation)
# df["date"] = df["Start_Time"].dt.date
df["hour"] = df["Start_Time"].dt.hour
df['day'] = df['Start_Time'].dt.day
df['month'] = df['Start_Time'].dt.month
df['dayofweek'] = df['Start_Time'].dt.dayofweek

# Map hour to coarse time-of-day bucket
df["time_bucket"] = df["hour"].apply(hour_to_time_bucket)
df["season"] = df["month"].apply(month_to_season)
df["is_rushhour"] = df["hour"].apply(is_rushhour)
us_holidays = holidays.US()
df['is_holiday'] = df['Start_Time'].dt.date.apply(lambda x: x in us_holidays)



# dropping original columns after feature engineering 
# Start_Time,Weather_Condition, Distance(mi)
df = df.drop(columns=["Start_Time", "Weather_Condition", "Distance(mi)", "Wind_Direction"])
df = df.dropna()
df = df.reset_index(drop=True)

# one hot encoding
# df = pd.get_dummies(df, drop_first=True)

# Downcast integer and float columns
for col in df.select_dtypes(include=['int64']).columns:
    df[col] = pd.to_numeric(df[col], downcast='integer')

for col in df.select_dtypes(include=['float64']).columns:
    df[col] = pd.to_numeric(df[col], downcast='float')

# Convert object columns to category (saves memory if many repeated values)
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].astype('category')

print("downcasted processed dataset")
# Finally, save to Parquet with compression
# 

print(df.info())
###print()
# print(df)
print(df.columns)


print("saving parquet", df.shape)
###print(df.columns)
# df.to_parquet("data/processed/US_Accidents_Cleaned.parquet", engine="pyarrow", compression="snappy")
df.to_parquet("data/processed/US_Accidents_Processed.parquet", engine="pyarrow", compression="snappy")
# df.to_csv("data/processed/US_Accidents_Cleaned.csv", index=False)
print("success")