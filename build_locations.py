import pandas as pd
import reverse_geocoder as rg

if __name__ == '__main__':
    print("Loading dataset to find unique coordinates...")
    df = pd.read_csv("Final_Master_Flood_Dataset_Corrected.csv", encoding='latin1', low_memory=False)

    # Get the unique locations and their average terrain data
    unique_locs = df.groupby(['Latitude', 'Longitude']).agg({
        'Elevation': 'mean',
        'Slope': 'mean',
        'Water_Occurrence_Percent': 'mean',
        'Population_Density': 'mean'
    }).reset_index()

    # FIX 1: Strip away Pandas formatting and force them to be raw Python floats
    coords = [(float(row['Latitude']), float(row['Longitude'])) for _, row in unique_locs.iterrows()]

    print(f"Found {len(coords)} unique coordinates. Snapping to nearest known cities...")

    # FIX 2: mode=1 forces single-thread processing. No more Windows crashes!
    results = rg.search(coords, mode=1)

    location_data = []

    for idx, row in unique_locs.iterrows():
        city_info = results[idx]
        
        district = city_info.get('admin2', '')
        city = city_info.get('name', 'Unknown City')
        
        if not district:
            district = city_info.get('admin1', 'Unknown State')
            
        location_data.append({
            'District': district,
            'City': city,
            'Latitude': row['Latitude'],
            'Longitude': row['Longitude'],
            'Elevation': row['Elevation'],
            'Slope': row['Slope'],
            'Water_Occurrence_Percent': row['Water_Occurrence_Percent'],
            'Population_Density': row['Population_Density']
        })
        print(f"Mapped: {row['Latitude']}, {row['Longitude']} -> {city}, {district}")

    # Save this cleaned dictionary
    loc_df = pd.DataFrame(location_data)
    loc_df = loc_df.sort_values(by=['District', 'City'])
    loc_df.to_csv("Location_Dictionary.csv", index=False)
    
    print("\n✅ Success! Saved all recognized locations to 'Location_Dictionary.csv'")