import pandas as pd

# Read the main CSV file
df = pd.read_csv('df_cleaned.csv')

# List of cities
cities = ['dallas', 'houston', 'la', 'nyc', 'philadelphia', 
          'phoenix', 'san_antonio', 'san_diego', 'san_jose', 'seattle']

# Split and save data for each city
for city in cities:
    # Filter data for current city
    city_df = df[df['city'] == city]
    
    # Save to a new CSV file
    output_file = f'{city}.csv'
    city_df.to_csv(output_file, index=False)
    print(f'Created {output_file}')