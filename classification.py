from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

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
    ('classifier', RandomForestClassifier(random_state=42))
])

# Train the model on the entire training dataset
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
report = classification_report(y_test, y_pred)
print(f'Report: {report}')
conf_matrix = confusion_matrix(y_test, y_pred)

# Save Results to a Text File in the 'reports' Folder
report_path = os.path.join(reports_path, 'imbalanced_random_forest_report.log')
with open(report_path, 'w') as f:
    f.write(f'Report: {report}')

# Display confusion matrix with original genre names
ConfusionMatrixDisplay(conf_matrix, display_labels=y_train.unique()).plot()
plt.title('Confusion Matrix for Genre Classification')
# Save the plot as an image
plt.savefig(os.path.join(figures_path, 'confusion_matrix_imbalanced_random_forest.png'), bbox_inches='tight')
plt.close()

# Overall Feature Importance

# Extract feature importances from the trained model
feature_importances = model.named_steps["classifier"].feature_importances_
transformed_feature_names = model.named_steps["preprocessor"].get_feature_names_out()

# Create a dictionary to map feature importance back to original features
feature_importance_dict = defaultdict(float)

for i, transformed_feature in enumerate(transformed_feature_names):
    if transformed_feature.startswith("num__"):  # Numerical features
        original_feature = transformed_feature.replace("num__", "")
        feature_importance_dict[original_feature] += feature_importances[i]
    elif transformed_feature.startswith("cat__"):  # Categorical features (One-Hot Encoded)
        original_feature = transformed_feature.split("__")[1].rsplit("_", 1)[0]
        feature_importance_dict[original_feature] += feature_importances[i]  # Aggregate importance

# Convert to Pandas Series and sort
importance_series = pd.Series(feature_importance_dict).sort_values(ascending=False)

# Plot overall feature importance
plt.figure(figsize=(10, 6))
plt.barh(importance_series.index, importance_series.values)
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Overall Feature Importance for the Imbalanced Random Forest Model")

# Save the figure
plt.savefig(os.path.join(figures_path, 'overall_feature_importance.png'), bbox_inches='tight')
plt.close()


# Feature Importance for Each Genre

# Original feature names
original_feature_names = numerical_cols.tolist() + categorical_cols.tolist()

# Initialize a dictionary to store feature importance for all genres
genre_importance_dict = {genre: {feature: 0 for feature in original_feature_names} for genre in df_train['tag'].unique()}

# Train a model for each genre and calculate feature importance
for genre in df_train['tag'].unique():
    # Create a binary target variable for the current genre
    y = (df_train['tag'] == genre).astype(int)
    
    # Define the model
    model = RandomForestClassifier(random_state=42)
    
    # Create a pipeline that preprocesses the data and then trains the model
    clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    
    # Fit the model
    clf.fit(X_train, y)
    
    # Get feature importances
    feature_importances = clf.named_steps['classifier'].feature_importances_
    transformed_feature_names = clf.named_steps['preprocessor'].get_feature_names_out()
    
    # Map feature importance back to original features
    for i, transformed_feature in enumerate(transformed_feature_names):
        if transformed_feature.startswith('num__'):
            original_feature = transformed_feature.replace('num__', '')
            genre_importance_dict[genre][original_feature] += feature_importances[i]
        elif transformed_feature.startswith('cat__'):
            original_feature = transformed_feature.split('__')[1].rsplit('_', 1)[0]
            genre_importance_dict[genre][original_feature] += feature_importances[i]

# Convert the dictionary to a DataFrame for visualization
genre_importance_df = pd.DataFrame.from_dict(genre_importance_dict, orient='index')

# Plot feature importance for each genre individually
for genre, importance_series in genre_importance_df.iterrows():
    # Sort the feature importances in descending order
    importance_series = importance_series.sort_values(ascending=False)
    
    # Create a new figure for each genre
    plt.figure(figsize=(10, 6))
    plt.barh(importance_series.index, importance_series.values)
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.title(f'Feature Importance for Genre: {genre}')
    
    # Save the plot as an image
    plt.savefig(os.path.join(figures_path, f'feature_importance_{genre}.png'), bbox_inches='tight')
    plt.close()

# Aggregate feature importances across all genres
overall_importance_dict = {feature: 0 for feature in original_feature_names}
for genre in genre_importance_dict:
    for feature in genre_importance_dict[genre]:
        overall_importance_dict[feature] += genre_importance_dict[genre][feature]

# Convert the overall importance dictionary to a DataFrame for visualization
overall_importance_df = pd.DataFrame({
    'Feature': list(overall_importance_dict.keys()),
    'Importance': list(overall_importance_dict.values())
})

# Sort by importance
overall_importance_df = overall_importance_df.sort_values(by='Importance', ascending=False)

# Plot aggregated feature importance
plt.figure(figsize=(10, 6))
plt.barh(overall_importance_df['Feature'], overall_importance_df['Importance'])
plt.xlabel('Aggregated Feature Importance')
plt.ylabel('Feature')
plt.title('Aggregated Feature Importance for Genre Classification')
plt.savefig(os.path.join(figures_path, 'aggregated_feature_importance.png'), bbox_inches='tight')
plt.close()