#Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Import dataset 
df = pd.read_csv('US_Accidents_Dec21_updated.csv')

#Add date columns
df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
df['End_Time'] = pd.to_datetime(df['End_Time'], errors='coerce')
df['Year']=df['Start_Time'].dt.year
df['Month']=df['Start_Time'].dt.strftime('%b')
df['Day']=df['Start_Time'].dt.day
df['Hour']=df['Start_Time'].dt.hour
df['Weekday']=df['Start_Time'].dt.strftime('%a')
td='Time_Duration(min)'
df[td]=round((df['End_Time']-df['Start_Time'])/np.timedelta64(1,'m'))

#Display map of US
x = 'Start_Lng'
y = 'Start_Lat'
plt.scatter(x,y,data=df,s=0.3)
plt.title("Map of accidents in the US")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()

#Display bar plot of accidents per state (Top 15)
x='State'
y=df['State'].value_counts()
plt.pyplot.bar(x,y,data=df)
plt.title("Top 10 states with the highest number of accidents")
plt.xlabel("Number of accident")
plt.ylabel("State")
plt.show()

#Display bar plot of top 15 cities with highest amount of road accidents
city_counts=df['City'].value_counts()
city_counts[:15].plot(kind='bar')
plt.xlabel("Cities")
plt.ylabel("Number of accidents")
plt.title("Top 15 cities with highest amount of road accidents")
plt.show()

#Display accidents per year
year_counts=df['Year'].value_counts()
year_counts.plot(kind='bar')
plt.xlabel("Year")
plt.ylabel("Number of accidents")
plt.title("Accidents per year")
plt.show()

#Display accidents per day of the week
day_counts=df['Weekday'].value_counts()
day_counts.plot(kind='bar')
plt.xlabel("Day of the week")
plt.ylabel("Number of accidents")
plt.title("Accidents per day of the week")
plt.show()

#Display accidents per month
month_counts=df['Month'].value_counts()
month_counts.plot(kind='bar')
plt.xlabel("Month")
plt.ylabel("Number of accidents")
plt.title("Accidents per month")
plt.show()

#Display comparison of accidents during day and during night
civil_counts=df['Civil_Twilight'].value_counts()
plt.pie(civil_counts,labels=civil_counts.index,autopct='%1.1f%%')
plt.title("Accidents in day and night")
plt.show()

#Display accidents per hour
plt.hist(df['Hour'],histtype='bar')
plt.xlabel("Hour")
plt.ylabel("Number of accidents")
plt.title("Accidents per hour")
plt.show()

#Display comparison of accidents per side of the road
side_counts=df['Side'].value_counts()
side=side_counts[:2]
plt.pie(side,labels=side.index,autopct='%1.1f%%')
plt.title("Accidents per side of the road")
plt.show()

#Display most common weather conditions during accidents
weather_counts = df["Weather_Condition"].value_counts()[:10]
plt.figure
plt.title("Most common weather conditions")
weather_counts.plot(kind='bar')
plt.xlabel("Weather Condition")
plt.ylabel("Number of accidents")
plt.show()

#Display most common road features present during accidents
features = ["Amenity", "Bump", "Crossing", "Give_Way", "Junction", "No_Exit", "Railway", "Roundabout", "Station", "Stop", "Traffic_Calming", "Traffic_Signal", "Turning_Loop"]
features_counts = df[features].sum().sort_values(ascending=False)
features_counts.plot(kind='bar')
plt.title("Most common features")
plt.xlabel("Features")
plt.ylabel("Number of accidents")
plt.show()

#Display severity accidents
severity_counts=df['Severity'].value_counts()
severity_counts.plot(kind='bar')
plt.xlabel("Severity")
plt.ylabel("Number of accidents")
plt.title("Severity of accidents")
plt.show()