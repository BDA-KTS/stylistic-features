from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, recall_score
import os
import pandas as pd
from sklearn.dummy import DummyClassifier

# Create the 'reports' folder if it doesn't exist
if not os.path.exists('reports'):
    os.makedirs('reports')

# Load datasets
df_train = pd.read_csv('datasets/train/processed_train_dataset.csv')

df_test = pd.read_csv('datasets/test/processed_test_dataset.csv')

# Step 1: Split data into features (X) and labels (y)
X_train = df_train[['repetitiveness', 'profanity', 'flesch_reading_scores', 'sentence_count', 'reading_time', 'char_count', 'monosyllable_count', 'polysyllable_count', 'syllable_count', 'text_standard', 'unique_tokens_count', 'ttr', 'rttr', 'cttr', 'herdan', 'maas', 'yulei', 'sentiment', 'emotions', 'number_of_different_rhyme_schemes', 'unique_rhyme_labels', 'rhyme_repetition_rate', 'rhyme_entropy', 'average_rhyme_length', 'topic_id']]  # Features
y_train = df_train['tag']  # Target variable

X_test = df_test[['repetitiveness', 'profanity', 'flesch_reading_scores', 'sentence_count', 'reading_time', 'char_count', 'monosyllable_count', 'polysyllable_count', 'syllable_count', 'text_standard', 'unique_tokens_count', 'ttr', 'rttr', 'cttr', 'herdan', 'maas', 'yulei', 'sentiment', 'emotions', 'number_of_different_rhyme_schemes', 'unique_rhyme_labels', 'rhyme_repetition_rate', 'rhyme_entropy', 'average_rhyme_length', 'topic_id']]  # Features
y_test = df_test['tag']  # Target variable

# Step 1: Train the Majority Class Model
majority_class_model = DummyClassifier(strategy='most_frequent', random_state=42)
majority_class_model.fit(X_train, y_train)

# Step 2: Predict using the Majority Class Model
y_majority_pred = majority_class_model.predict(X_test)

# Step 3: Evaluate the Majority Class Model
majority_report = classification_report(y_test, y_majority_pred, zero_division=0)  # Avoid division errors if a class is missing

# Step 4: Save Results to a Text File in the 'reports' Folder
report_path = os.path.join('reports', 'majority_class_report.log')
with open(report_path, 'w') as f:
    f.write(f'Report: {majority_report}')

# print the report to the console
print(f'Report: {majority_report}')