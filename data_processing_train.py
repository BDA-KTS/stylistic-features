import pandas as pd
import repetitiveness as rep
import os
import lexicalanalysis
import sentiments_and_emotions as emo
import rhyme_scheme as rhyme
import profanity_analysis as profanity
import readability
import bertopic_modeling as bm

# Load the data
df = pd.read_csv('datasets/train/preprocessed_train_dataset.csv')

# Drop the None values
df = df.dropna()

# Resetting the index
df = df.reset_index(drop=True)

# Train and export bertopic model
training_model = bm.BertopicTraining(df['punctuated_text'])
training_model.save("Bertopic")

# Helper function to apply function safely
def safe_apply(func, value, default=None):
    try:
        return func(value)
    except Exception as e:
        print(f"Error processing {func.__name__} for value {value}: {e}")
        return default

#Repetitiveness
df['repetitiveness'] = df['lyrics'].map(lambda x: safe_apply(rep.num_of_choruses, x))

# Profanity
df['profanity'] = df['clean_lyrics'].map(lambda x: safe_apply(profanity.count_profanity, x))

# Readability
df['flesch_reading_scores'] = df['punctuated_text'].map(lambda x: safe_apply(readability.get_flesch_reading_ease, x))
df['sentence_count'] = df['punctuated_text'].map(lambda x: safe_apply(readability.get_sentence_count, x))
df['reading_time'] = df['punctuated_text'].map(lambda x: safe_apply(readability.get_reading_time, x))
df['char_count'] = df['punctuated_text'].map(lambda x: safe_apply(readability.get_char_count, x))
df['monosyllable_count'] = df['punctuated_text'].map(lambda x: safe_apply(readability.get_monosyllable_count, x))
df['polysyllable_count'] = df['punctuated_text'].map(lambda x: safe_apply(readability.get_polysyllable_count, x))
df['syllable_count'] = df['punctuated_text'].map(lambda x: safe_apply(readability.get_syllable_count, x))
df['text_standard'] = df['punctuated_text'].map(lambda x: safe_apply(readability.get_text_standard, x))

# Lexical Analysis
df['unique_tokens_count'] = df['clean_lyrics'].map(lambda x: safe_apply(lexicalanalysis.get_tokens, x))
df['ttr'] = df['clean_lyrics'].map(lambda x: safe_apply(lexicalanalysis.get_ttr, x))
df['rttr'] = df['clean_lyrics'].map(lambda x: safe_apply(lexicalanalysis.get_rttr, x))
df['cttr'] = df['clean_lyrics'].map(lambda x: safe_apply(lexicalanalysis.get_cttr, x))
df['herdan'] = df['clean_lyrics'].map(lambda x: safe_apply(lexicalanalysis.get_herdan, x))
df['maas'] = df['clean_lyrics'].map(lambda x: safe_apply(lexicalanalysis.get_maas, x))
df['yulei'] = df['clean_lyrics'].map(lambda x: safe_apply(lexicalanalysis.get_yulei, x))


# Emotion Analysis
df['sentiment'] = df['punctuated_text'].map(lambda x: safe_apply(emo.get_vader_scores, x))
df['emotions'] = df['clean_lyrics'].map(lambda x: safe_apply(emo.emotion_classification, x))

# Rhyme Schemes
df[['number_of_different_rhyme_schemes', 'unique_rhyme_labels', 'rhyme_repetition_rate', 'rhyme_entropy', 'average_rhyme_length']] = df['lyrics'].map(lambda x: safe_apply(rhyme.process_rhyme_metrics, x)).apply(pd.Series)

# Topics
df["topic_id"] = training_model.topics

df = df.dropna()

# Export the results
df.to_csv('datasets/train/processed_train_dataset.csv', index=False)