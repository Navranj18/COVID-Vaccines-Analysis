# COVID-Vaccines-Analysis
DATA PREPROCESSING
   Data preprocessing, in the context of data analysis and machine learning, refers to the techniques and procedures used to clean, transform, and organize raw data into a format suitable for analysis. It is a crucial step in the data preparation process and is essential for ensuring that the data used for analysis or modeling is accurate, consistent, and relevant.

CODE
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
# Load COVID vaccine data (replace 'covid_vaccine_data.csv' with your data file)
data_file = "country_vaccinations.csv"
covid_vaccine_data = pd.read_csv(data_file)
# Remove duplicates (if applicable)
covid_vaccine_data.drop_duplicates(inplace=True)
# Fill missing values with the mean of each column
covid_vaccine_data.fillna(covid_vaccine_data.mean(), inplace=True)
# Convert the 'timestamp' column to datetime (if applicable)
covid_vaccine_data['timestamp'] = pd.to_datetime(covid_vaccine_data['timestamp'])
# Compute daily averages (adjust columns accordingly)
daily_averages = covid_vaccine_data.groupby(covid_vaccine_data['timestamp'].dt.date).mean()
# Scale the data using Min-Max scaling (adjust columns accordingly)
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(covid_vaccine_data[['feature1', 'feature2']])
# Split the data into training and testing sets
X = covid_vaccine_data.drop(columns=['target_column']) # Adjust 'target_column' to your target
variable
y = covid_vaccine_data['target_column']
# Adjust 'target_column' to your target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Save daily averages to a CSV file (if needed)
daily_averages.to_csv("preprocessed_country_vaccinations.csv", index=False)
DATA ANALYSIS
Analysis refers to the process of examining, evaluating, and interpreting data, information, or a complex subject to gain a deeper understanding, draw insights, and make informed decisions. It is a systematic and methodical approach to breaking down a problem, situation, or dataset into its constituent parts, investigating relationships, patterns, and trends, and drawing meaningful conclusions.
CODE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv("country_vaccinations.csv")
data.head()
data.describe()
pd.to_datetime(data.date)
data.country.value_counts()
data = data[data.country.apply(lambda x: x not in ["England", "Scotland", "Wales", "Northern Ireland"])]
data.country.value_counts()
data.vaccines.value_counts()
df = data[["vaccines", "country"]]
df.head()
dict_ = {}
for i in df.vaccines.unique():
  dict_[i] = [df["country"][j] for j in df[df["vaccines"]==i].index]
 vaccines = {}
for key, value in dict_.items():
  vaccines[key] = set(value)
for i, j in vaccines.items():
  print(f"{i}:>>{j}")

Let’s visualize this data to have a look at what combination of vaccines every country is using:

import plotly.express as px
import plotly.offline as py
 vaccine_map = px.choropleth(data, locations = 'iso_code', color = 'vaccines')
vaccine_map.update_layout(height=300, margin={"r":0,"t":0,"l":0,"b":0})
vaccine_map.show()

NOTE
All the above code(s) can be executed using Jupyter notebook.
