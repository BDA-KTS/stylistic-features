import pandas as pd
from sklearn.model_selection import train_test_split
import os

df = pd.read_csv('datasets/preprocessed_dataset.csv')

train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

# Save train and test datasets to CSV
os.makedirs("datasets/train/", exist_ok=True)
os.makedirs("datasets/test/", exist_ok=True)

train_data.to_csv('datasets/train/preprocessed_train_dataset.csv', index=False)
test_data.to_csv('datasets/test/preprocessed_test_dataset.csv', index=False)