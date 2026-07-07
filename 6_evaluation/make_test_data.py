import pandas as pd

# Load the dataset
file_path = "D:/1002/combined_sorted_track_history_11_to_50_1.csv"

df = pd.read_csv(file_path)

# Display the first few rows to understand the structure of the dataset
df.head()


# Filtering the dataset based on the provided StationID and TrackID
outlier_station_ids = {
    24: [396, 29, 450, 495, 183, 181, 177],
    31: [471],
    38: [409, 265],
    49: [91, 62]
}

filtered_df_combined = df[
    (df['StationID'] == 24) & (df['TrackID'].isin([396, 29, 450, 495, 183, 181, 177])) |
    (df['StationID'] == 31) & (df['TrackID'].isin([471])) |
    (df['StationID'] == 38) & (df['TrackID'].isin([409, 265])) |
    (df['StationID'] == 49) & (df['TrackID'].isin([91, 62]))
]

# Display filtered dataframe to user
filtered_df_combined.to_csv("D:/1002/test_data.csv")
