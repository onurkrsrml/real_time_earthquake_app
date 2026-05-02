import reverse_geocoder as rg
import pandas as pd

def coordinate_finder(dataframe, lat_col, long_col):
    # Create coordinate tuples from the dataframe columns
    coordinates = list(zip(dataframe[lat_col], dataframe[long_col]))

    # Perform batch search
    results = rg.search(coordinates)

    # Convert results to a dataframe and join with the original
    location_df = pd.DataFrame(results)
    
    # Rename columns for clarity in the earthquake context
    location_df = location_df.rename(columns={
        'admin1': 'state',
        'name': 'city_name'
    })

    # Keep only the requested columns (state and city_name)
    location_df = location_df[['state', 'city_name']]

    # Reset indices to ensure proper alignment during concatenation
    dataframe = dataframe.reset_index(drop=True)
    location_df = location_df.reset_index(drop=True)

    df_coordinated = pd.concat([dataframe, location_df], axis=1)

    return df_coordinated
