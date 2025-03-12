import re
import numpy as np
import pandas as pd

def clean_text(text):
    # Remove metadata within square brackets
    text = re.sub(r'\[.*?\]', "", text) 
    # Convert text to lowercase
    text = text.lower()
    return text

# Load the dataset
lyric_database = pd.read_csv("datasets/songs_dataset.csv")

# Filter out rows with infinite values in numerical columns
numerical_cols = lyric_database.select_dtypes(include=['int64', 'float64']).columns

pattern = r'\[.*?\]'

df = (
    lyric_database
    .loc[lyric_database["language"] == "en"]
    .loc[lyric_database["tag"] != "misc"]
    .loc[lyric_database["tag"] != ""]
    .loc[lyric_database["tag"] != None]
    .loc[lyric_database["lyrics"] != None]
    .loc[lyric_database["lyrics"] != ""]
    .loc[lyric_database["lyrics"] != "[Instrumental]"]
    .loc[lyric_database["lyrics"].str.contains(pattern)]
    .loc[~np.isinf(lyric_database[numerical_cols]).any(axis=1)]
)

# Cleaning the lyrics
df['clean_lyrics'] = df['lyrics'].map(clean_text)

# Removing irrelevant features
df = df[['artist', 'title', 'year', 'tag', 'lyrics', 'clean_lyrics']]

# Save the clean dataset
df.to_csv("datasets/clean_dataset.csv", index=False)