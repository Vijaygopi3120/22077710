#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This script analyzes GDP growth and CO2 emissions data from World Bank datasets.

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcdefaults()
from scipy.stats import linregress

import warnings
warnings.filterwarnings("ignore")


# Read GDP growth dataset
gdp_growth = pd.read_csv("GDP_growth.csv", skiprows=4)
# Drop columns with NaN values
gdp_growth = gdp_growth.dropna(axis=1, how='all')
gdp_growth.head()

# Read CO2 emissions dataset
CO2 = pd.read_csv("CO2_emissions.csv", skiprows=4)
# Drop columns with NaN values
CO2 = CO2.dropna(axis=1, how='all')
CO2.head()


def process_world_bank_data(filename):
    
    """
    Reads a World Bank dataset file, drops NaN columns, and transposes the dataframe.

    Parameters:
    - filename (str): The path to the World Bank dataset file.

    Returns:
    - df_cleaned (pd.DataFrame): The cleaned dataframe.
    - df_transposed (pd.DataFrame): The transposed dataframe.
    """
    
    # Read the dataframe from the given filename
    df = pd.read_csv(filename, skiprows=4)

    # Drop columns with NaN values
    df_cleaned = df.dropna(axis=1, how='all')

    # Transpose the dataframe
    df_transposed = df_cleaned.transpose()

    # Set the first row as column headers
    df_transposed.columns = df_transposed.iloc[0]

    # Remove the first row (header row)
    df_transposed = df_transposed[1:]

    return df_cleaned, df_transposed

# Example usage for GDP growth dataset
gdp_growth_cleaned, gdp_growth_transposed = process_world_bank_data("GDP_growth.csv")
print("GDP Growth Cleaned DataFrame:")
print(gdp_growth_cleaned.head())
print("\nGDP Growth Transposed DataFrame:")
print(gdp_growth_transposed.head())

# Example usage for CO2 emissions dataset
CO2_cleaned, CO2_transposed = process_world_bank_data("CO2_emissions.csv")
print("\nCO2 Emissions Cleaned DataFrame:")
print(CO2_cleaned.head())
print("\nCO2 Emissions Transposed DataFrame:")
print(CO2_transposed.head())




#Converting the different year columns into one year column with the different years in them.
gdp_growth_melted = pd.melt(gdp_growth, id_vars=['Country Name', 'Country Code'], var_name='Year', value_name='GDP growth (annual %)')
CO2_melted = pd.melt(CO2, id_vars=['Country Name', 'Country Code'], var_name='Year', value_name='CO2 emissions (kt)')

# Merge dataframes based on 'Country Name', 'Country Code', and 'Year'
merged_df = pd.merge(gdp_growth_melted, CO2_melted, on=['Country Name', 'Country Code', 'Year'])

# Drop rows with NaN values
merged_df = merged_df.dropna()
merged_df.head()
print(merged_df.columns)
print(merged_df.dtypes)

# Filter out non-numeric values in 'Year' column
merged_df = merged_df[merged_df['Year'].apply(lambda x: str(x).isdigit())]
# Convert 'Year' column to integer
merged_df['Year'] = merged_df['Year'].astype(int)

# Convert 'GDP growth (annual %)' column to float
merged_df['GDP growth (annual %)'] = pd.to_numeric(merged_df['GDP growth (annual %)'], errors='coerce')

# Convert 'CO2 emissions (kt)' column to float
merged_df['CO2 emissions (kt)'] = pd.to_numeric(merged_df['CO2 emissions (kt)'], errors='coerce')

print(merged_df.dtypes)

# Selecting a country
selected_country = 'China'
selected_country_data = merged_df[merged_df['Country Name'] == selected_country]

# Explore the correlation between CO2 emissions and GDP for the selected country
correlation = selected_country_data['GDP growth (annual %)'].corr(selected_country_data['CO2 emissions (kt)'])
print(f"Correlation between GDP growth and CO2 emissions for {selected_country}: {correlation}")

# Visualizations
# Bar chart for GDP growth and CO2 emissions for the selected country
#First turn the year data into year range 
selected_country_data['Year Range'] = pd.cut(selected_country_data['Year'], bins=range(1960, 2030, 10), right=False)

# Melt the DataFrame to have a single 'Value' column and 'Indicator' column
melted_data = pd.melt(selected_country_data, id_vars=['Year Range'], value_vars=['GDP growth (annual %)', 'CO2 emissions (kt)'], var_name='Indicator', value_name='Value')

# Grouped bar chart for GDP growth and CO2 emissions for the selected country
plt.figure(figsize=(12, 6))
sns.barplot(x='Year Range', y='Value', hue='Indicator', data=melted_data, palette={'GDP growth (annual %)': 'blue', 'CO2 emissions (kt)': 'red'})
plt.title(f'GDP Growth vs CO2 Emissions for {selected_country}')
plt.xlabel('Year Range')
plt.ylabel('Value')
plt.legend()
plt.show()

# Heat map for correlation between CO2 emissions and GDP growth
plt.figure(figsize=(8, 6))
sns.heatmap(selected_country_data[['GDP growth (annual %)', 'CO2 emissions (kt)']].corr(), annot=True, cmap='coolwarm')
plt.title(f'Correlation Heatmap for {selected_country}')
plt.show()



# Perform linear regression
slope, intercept, r_value, p_value, std_err = linregress(selected_country_data['GDP growth (annual %)'], selected_country_data['CO2 emissions (kt)'])

# Visualizations
# Scatter plot with regression line
plt.figure(figsize=(12, 6))
sns.scatterplot(x='GDP growth (annual %)', y='CO2 emissions (kt)', data=selected_country_data, label=f'{selected_country} Data', alpha=0.7)
plt.plot(selected_country_data['GDP growth (annual %)'], intercept + slope * selected_country_data['GDP growth (annual %)'], color='red', label='Regression Line')
plt.title(f'Scatter Plot with Regression Line for {selected_country}')
plt.xlabel('GDP Growth (annual %)')
plt.ylabel('CO2 Emissions (kt)')
plt.legend()
plt.show()


print(merged_df.head())

#Finding relevant data from some selected and important countries
selected_countries = ['Australia', 'India', 'Bangladesh', 'China', 'United States']
selected_years = [1990, 2005, 2010, 2014]

selected_data = merged_df.loc[(merged_df['Country Name'].isin(selected_countries)) & (merged_df['Year'].isin(selected_years))]
print(selected_data.head(20))


table_data = selected_data.pivot(index='Country Name', columns='Year', values=['GDP growth (annual %)', 'CO2 emissions (kt)'])
print(table_data)
table_data.to_csv('output_table.csv', index=True)


# Visualisation for GDP rise in different countries
plt.figure(figsize=(12, 6))
palette = sns.color_palette("coolwarm", len(selected_years))

# Plot the bar chart with side-by-side bars
for i, year in enumerate(selected_years):
    year_data = selected_data[selected_data['Year'] == year]
    plt.bar(
        np.arange(len(selected_countries)) + i * 0.2,
        year_data['GDP growth (annual %)'],
        width=0.2,
        label=str(year),
        color=palette[i]
    )

plt.title('GDP Growth Over Time for Selected Countries')
plt.xlabel('Country')
plt.ylabel('GDP Growth (annual %)')
plt.xticks(np.arange(len(selected_countries)) + 0.3, selected_countries)
plt.legend(title='Year')

plt.show()



# Visualisation for rise in carbon emissions in different countries
plt.figure(figsize=(12, 6))
custom_palette = ['green', 'red', 'purple', 'black']

# Plot the bar chart with side-by-side bars for CO2 emissions
for i, year in enumerate(selected_years):
    year_data = selected_data[selected_data['Year'] == year]
    plt.bar(
        np.arange(len(selected_countries)) + i * 0.2,
        year_data['CO2 emissions (kt)'],
        width=0.2,
        label=str(year),
        color=custom_palette[i]
    )

plt.title('CO2 Emissions Over Time for Selected Countries')
plt.xlabel('Country')
plt.ylabel('CO2 Emissions (kt)')
plt.xticks(np.arange(len(selected_countries)) + 0.3, selected_countries)
plt.legend(title='Year')

plt.show()
