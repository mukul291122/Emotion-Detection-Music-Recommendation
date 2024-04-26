import pandas as pd

# Define the path to the CSV file
csv_path = "C:/Users/Vaidehi Suryawanshi/Downloads/Music_Face1/Music_Face/spotify-music-data-to-identify-the-moods/data_moods.csv"

# Load music data
music_data = pd.read_csv(csv_path)

# Print column names
print(music_data.columns)
