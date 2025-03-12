# A Lyric-Driven Approach to Genre Music Classification

This repository contains the code and resources for the thesis **"A Lyric-Driven Approach to Genre Music Classification"** by **Omar Chouikha**. The research explores the use of song lyrics as a primary feature for classifying music genres. By leveraging natural language processing (NLP) techniques and machine learning models, the study aims to demonstrate the effectiveness of lyrical content in distinguishing between different music genres.

## Summary of the Research

The research focuses on analyzing and processing song lyrics to extract meaningful features that can be used for genre classification. The pipeline includes data cleaning, punctuation restoration, dataset splitting, and feature extraction. The final step involves training and evaluating various classification models to predict music genres based on lyrical content.

## Datasets

All relevant datasets are available on [Zenodo](https://zenodo.org/records/14969295).

To replicate the entire pipeline, you will need to download the raw dataset:

- **Original Dataset:**  
  Download from [Kaggle](https://www.kaggle.com/datasets/carlosgdcj/genius-song-lyrics-with-language-information).  
  This dataset is required if you want to replicate the entire pipeline, starting from raw data.

  **NOTE:** Make sure to store it in a folder named `datasets`

## Pipeline Steps

1. **Data Cleaning**  
   To clean the dataset and prepare it for further processing, run the following script:

    ```bash
    python dataset_cleaning.py

This will generate a `clean_dataset.csv` file containing the cleaned version of the dataset.

2. **Restoring Punctuation**  
   To restore punctuation in the dataset, execute the following:

   ```bash
    python punctuation.py

This will generate the `preprocessed_dataset.csv` file, which contains the dataset with restored punctuation.

3. **Splitting the Dataset**  
   To split the dataset into training and testing sets, run:

   ```bash
   python dataset_splitting.py

This generates the processed datasets: `preprocessed_train_dataset.csv` and `preprocessed_test_dataset.csv`

4. **Processing**  
   Now, you need to process the training and test datasets. Run these script **in this exact order**:

   ```bash
   python data_processing_train.py
   python data_processing_test.py

This generates the processed datasets: `processed_train_dataset.csv` and `processed_test_dataset.csv`

5. **Classification**  
   Run the different classification scripts to train and evaluate the models.  
   These scripts will use the processed datasets to predict music genres based on lyrical content.
   All reports will be saved in a dedicated folder named `reports`, with figures stored in a corresponding subfolder located at `reports/figures`.

## Usage

To replicate the pipeline, follow the steps above in the specified order. Ensure all required datasets are downloaded and placed in the correct directory before executing the scripts. All datasets should be stored in a folder named `datasets`.

**Note:** The following steps are only relevant if you are skipping directly to the **classification** step:

1. Download the `processed_test_dataset.csv` and store it in a subfolder called `/datasets/test`.
2. Download the `processed_train_dataset.csv` and store it in a subfolder called `/datasets/train`.
3. Run any of the classification scripts (e.g., `classification.py`).

Install the dependencies using the following command:
```bash
pip install -r requirements.txt