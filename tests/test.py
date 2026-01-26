import pandas as pd
# pd.set_option("display.max_columns", None)

print("loading dataset")
# df = pd.read_csv("../us_accident_severity/data/processed/US_Accidents_Cleaned.csv", nrows=10000)
df = pd.read_parquet("../us_accident_severity/data/processed/US_Accidents_Cleaned.parquet")
print("loaded dataset")

print(df.columns)
print(df.head())
print(df.info())
print(df['Severity'].value_counts(normalize=True)*100)