from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import os
import pandas as pd

# Load datasets
df_train = pd.read_csv('datasets/train/processed_train_dataset.csv')

df_test = pd.read_csv('datasets/test/processed_test_dataset.csv')

# Create the folders if they don't exist
reports_path = 'reports'
if not os.path.exists(reports_path):
    os.makedirs(reports_path)    

# Step 1: Split data into features (X) and labels (y)
X_train = df_train[['repetitiveness', 'profanity', 'flesch_reading_scores', 'sentence_count', 'reading_time', 'char_count', 'monosyllable_count', 'polysyllable_count', 'syllable_count', 'text_standard', 'unique_tokens_count', 'ttr', 'rttr', 'cttr', 'herdan', 'maas', 'yulei', 'sentiment', 'emotions', 'number_of_different_rhyme_schemes', 'unique_rhyme_labels', 'rhyme_repetition_rate', 'rhyme_entropy', 'average_rhyme_length', 'topic_id']]  # Features
y_train = df_train['tag']  # Target variable

X_test = df_test[['repetitiveness', 'profanity', 'flesch_reading_scores', 'sentence_count', 'reading_time', 'char_count', 'monosyllable_count', 'polysyllable_count', 'syllable_count', 'text_standard', 'unique_tokens_count', 'ttr', 'rttr', 'cttr', 'herdan', 'maas', 'yulei', 'sentiment', 'emotions', 'number_of_different_rhyme_schemes', 'unique_rhyme_labels', 'rhyme_repetition_rate', 'rhyme_entropy', 'average_rhyme_length', 'topic_id']]  # Features
y_test = df_test['tag']  # Target variable

# Identify categorical and numerical columns
categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns
numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_cols),  # Numerical features (no transformation)
        ('cat', categorical_transformer, categorical_cols)  # Categorical features
    ])

# Append classifier to preprocessing pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DummyClassifier(strategy='uniform', random_state=42))
])

# Train the model on the entire training dataset
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

report = classification_report(y_test, y_pred)

print(f'Report: {report}')

# Save Results
report_path = os.path.join(reports_path, 'uniform_classifier_report.log')
with open(report_path, 'w') as f:
    f.write(f'Report: {report}')