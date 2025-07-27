# English-translated and annotated version of the original Pandas tutorial

import pandas as pd
import numpy as np
from enum import Enum

""" Introduction to Pandas DataFrames """
# Inspired by Keith Galli's excellent data analysis tutorial:
# https://www.youtube.com/watch?v=2uvysYbKdjM

# Creating a simple DataFrame with custom row and column labels
df = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=["A", "B", "C"],
                  index=["X", "Y", "Z"])

print(df.head())           # First 5 rows
print(df.head(1))          # First row
print(df.tail(1))          # Last row
print(df.columns)          # Column names
print(df.index)            # Index labels
print(df.info())           # Info summary (dtypes, non-null, etc.)
print(df.describe())       # Descriptive stats
print(df.shape)            # Shape (rows, columns)
print(df.size)             # Total elements
print(df.nunique())        # Unique values per column
print(df['A'].nunique())   # Unique values in column A

""" Reading DataFrames from files """
csv_file = pd.read_csv(
    "https://raw.githubusercontent.com/KeithGalli/complete-pandas-tutorial/refs/heads/master/warmup-data/coffee.csv")

print(csv_file.sample(5))
print(csv_file.loc[0])                # Access by index
print(csv_file.loc[[0, 2, 1]])        # Multiple rows
print(csv_file.loc[:4])               # Slice rows
print(csv_file.loc[5:8, ['Day']])     # Rows 5–8, column 'Day'
print(csv_file.iloc[:, [0, 2]])       # All rows, columns by position
print(csv_file.loc["Monday": "Wednesday", "Units Sold"])
csv_file.loc[1:3, "Units Sold"] = 10 # Modify specific values
print(csv_file)
print(csv_file.at[0, "Units Sold"])  # Fast scalar access
print(csv_file.iat[0, 0])             # Fast scalar access by position

""" Accessing data """
print(csv_file.Day)  # Access column (only if no spaces in name)
print(csv_file.sort_values("Units Sold", ascending=False))
print(csv_file.sort_values(["Units Sold", "Coffee Type"], ascending=[False, True]))

""" Filtering data """
bios = pd.read_csv(
    "https://raw.githubusercontent.com/KeithGalli/complete-pandas-tutorial/refs/heads/master/data/bios.csv")

print(bios[bios['height_cm'] > 215][['name', 'height_cm']])
print(bios[(bios['height_cm'] > 215) & (bios['born_country'] == 'BRA')])
print(bios[bios['name'].str.contains('keith', case=False)])
print(bios[bios['name'].str.contains('keith|patrick', case=False, regex=False)])
print(bios[bios['born_country'].isin(['FRA', 'GBR']) & bios['name'].str.startswith('Ke')]['name'])
print(bios.query('born_country == "USA" and born_city == "Seattle"'))

""" Adding and removing columns """
cafe = csv_file.copy()
cafe['price'] = 4.99
cafe['new_price'] = np.where(cafe['Coffee Type'] == 'Espresso', 3.99, 5.99)
print(cafe)
cafe.drop(0)  # Remove row (copy returned)
cafe.drop(columns=['price'], inplace=True)
print(cafe)

cafe['revenue'] = cafe['Units Sold'] * cafe['new_price']
cafe.rename(columns={'revenue': 'new_price'}, inplace=True)

bios_copy = bios.copy()
bios_copy['first_name'] = bios_copy['name'].str.split(' ').str[0]
print(bios_copy.query('first_name == "Keith"'))

bios_copy['born_datetime'] = pd.to_datetime(bios_copy['born_date'])
bios_copy['born_year'] = bios_copy['born_datetime'].dt.year

""" Custom column via apply() """
bios['height_category'] = bios['height_cm'].apply(
    lambda x: 'Short' if x < 165 else ('Average' if x < 185 else 'Tall'))

def categorize_athlete(row):
    if row['height_cm'] < 175 and row['weight_kg'] < 70:
        return 'lightweight'
    elif row['height_cm'] < 185 or row['weight_kg'] <= 80:
        return 'middleweight'
    else:
        return 'heavyweight'

bios['Category'] = bios.apply(categorize_athlete, axis=1)

""" Merging and concatenating data """
nocs = pd.read_csv(
    "https://raw.githubusercontent.com/KeithGalli/complete-pandas-tutorial/refs/heads/master/data/noc_regions.csv")

merged = pd.merge(bios, nocs, left_on='born_country', right_on='NOC', how='left', suffixes=['_x', '_y'])
merged.rename(columns={'region': 'born_country_full'}, inplace=True)
print(merged[merged['NOC_x'] != merged['born_country_full']][['name', 'NOC_x', 'born_country_full']])

usa = bios[bios['born_country'] == 'USA'].copy()
gbr = bios[bios['born_country'] == 'GBR'].copy()
combined = pd.concat([usa, gbr])

""" Handling missing values """
cafe.loc[[2, 3], 'Units Sold'] = np.nan
print(cafe.fillna(10))            # Replace NaNs with 10
print(cafe.interpolate())         # Fill using interpolation
print(cafe.dropna(subset=['Units Sold']))  # Drop rows with NaN in specific column
print(cafe[cafe['Units Sold'].notna()])    # Filter non-NaN
print(cafe[cafe['Units Sold'].isna()])     # Filter NaN

""" Aggregating data """
print(cafe.groupby(['Coffee Type'])['Units Sold'].mean())
print(cafe.groupby(['Coffee Type']).agg({'Units Sold': 'sum', 'price': 'mean'}))

pivot_df = cafe.pivot(columns='Coffee Type', index='Day', values='revenue')
print(pivot_df.sum(axis=1))

class Months(Enum):
    JANUARY = 1
    FEBRUARY = 2
    MARCH = 3
    APRIL = 4
    MAY = 5
    JUNE = 6
    JULY = 7
    AUGUST = 8
    SEPTEMBER = 9
    OCTOBER = 10
    NOVEMBER = 11
    DECEMBER = 12

bios['born_date'] = pd.to_datetime(bios['born_date'])
bios['month_born'] = bios['born_date'].dt.month.apply(
    lambda x: Months(x).name if pd.notnull(x) else None)
bios['year_born'] = bios['born_date'].dt.year

birth_counts = bios.groupby(['year_born', 'month_born'])['name'].count()
print(birth_counts.reset_index().sort_values('name', ascending=False))

""" Advanced functionality """
cafe['yesterday_revenue'] = cafe['revenue'].shift(1)
cafe['pct_change'] = cafe['revenue'] / cafe['yesterday_revenue'] * 100

bios['height_rank'] = bios['height_cm'].rank(ascending=False)
print(bios.sort_values(['height_rank']))

cafe['cumulative_revenue'] = cafe['revenue'].cumsum()
