from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import os

# Load datasets
df_train = pd.read_csv('datasets/train/processed_train_dataset.csv')

df_test = pd.read_csv('datasets/test/processed_test_dataset.csv')

# Create the folders if they don't exist
reports_path = 'reports'
if not os.path.exists(reports_path):
    os.makedirs(reports_path)    

figures_path = os.path.join(reports_path, 'figures')
if not os.path.exists(figures_path):
    os.makedirs(figures_path)

X_train = df_train[['repetitiveness', 'profanity', 'flesch_reading_scores', 'sentence_count', 'reading_time', 'char_count', 'monosyllable_count', 'polysyllable_count', 'syllable_count', 'text_standard', 'unique_tokens_count', 'ttr', 'rttr', 'cttr', 'herdan', 'maas', 'yulei', 'sentiment', 'emotions', 'number_of_different_rhyme_schemes', 'unique_rhyme_labels', 'rhyme_repetition_rate', 'rhyme_entropy', 'average_rhyme_length', 'topic_id']]  # Features
y_train = df_train['tag']

X_test = df_test[['repetitiveness', 'profanity', 'flesch_reading_scores', 'sentence_count', 'reading_time', 'char_count', 'monosyllable_count', 'polysyllable_count', 'syllable_count', 'text_standard', 'unique_tokens_count', 'ttr', 'rttr', 'cttr', 'herdan', 'maas', 'yulei', 'sentiment', 'emotions', 'number_of_different_rhyme_schemes', 'unique_rhyme_labels', 'rhyme_repetition_rate', 'rhyme_entropy', 'average_rhyme_length', 'topic_id']]  # Features
y_test = df_test['tag']

# Identify categorical and numerical columns
categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns
numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns

undersampler = RandomUnderSampler(random_state=42)

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

# Use Imbalanced Pipeline to handle resampling and preprocessing
balanced_pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('resampler', undersampler),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Train the model on the entire training dataset
balanced_pipeline.fit(X_train, y_train)

# Predict on the test set
y_pred = balanced_pipeline.predict(X_test)

# Step 6: Evaluate the model
report = classification_report(y_test, y_pred)
print(f'Report: {report}')

# Save Results
report_path = os.path.join(reports_path, 'balanced_random_forest_report.log')
with open(report_path, 'w') as f:
    f.write(f'Report: {report}')

conf_matrix = confusion_matrix(y_test, y_pred)    

# Display confusion matrix with original genre names
ConfusionMatrixDisplay(conf_matrix, display_labels=np.unique(y_test)).plot()
plt.title('Confusion Matrix for the Balanced Random Forest Classifier')
# Save the plot as an image
plt.savefig(os.path.join(figures_path, 'confusion_matrix_balanced_random_forest.png'), bbox_inches='tight')
plt.close()