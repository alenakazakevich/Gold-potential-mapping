import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('Sites.csv', encoding = "ISO-8859-1")
temp = pd.value_counts(df['TectonicUnit'])
names = ['Latitude', 'Longitude', 'Easting', 'Northing', 'Commodities']
data = df[names].dropna()

i=0
index = np.arange(len(data)).tolist()
# while i != index[-1]:




output = data['Commodities'].str.contains('Au', case=False)
