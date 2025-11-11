from features_util.lexicalanalysis import *
from features_util.profanity_analysis import *
from features_util.readability import *
from features_util.repetitiveness import *
from features_util.rhyme_scheme import *
from features_util.sentiments_and_emotions import *

import pandas as pd

if __name__ == "__main__":
    # Lexical analysis
    df = pd.read_csv("data/input_social_posts.csv")
    df['tokens'] = df['Posts'].apply(get_tokens)
    df['ttr'] = df['Posts'].apply(get_ttr)
    df['rttr'] = df['Posts'].apply(get_rttr)
    df['cttr'] = df['Posts'].apply(get_cttr)
    df['herdan'] = df['Posts'].apply(get_herdan)
    df['maas'] = df['Posts'].apply(get_maas)
    #df['yulei'] = df['Posts'].apply(get_yulei)

    # profanity features
    df['profanity_count'] = df['Posts'].apply(count_profanity)
    df['profanity_cleaned_text'] = df['Posts'].apply(clean_document) 

    # readability
    df['flesch_reading_ease'] = df['Posts'].apply(get_flesch_reading_ease)
    df['text_standard'] = df['Posts'].apply(get_text_standard)
    df['reading_time'] = df['Posts'].apply(get_reading_time)
    df['syllable_count'] = df['Posts'].apply(get_syllable_count)
    df['sentence_count'] = df['Posts'].apply(get_sentence_count)
    df['char_count'] = df['Posts'].apply(get_char_count)
    df['polysyllable_count'] = df['Posts'].apply(get_polysyllable_count)
    df['get_monosyllable_count'] = df['Posts'].apply(get_monosyllable_count)
    
    # repetitiveness
    df['num_of_choruses'] = df['Posts'].apply(num_of_choruses)

    # rhyme
    df['rhyme_scheme'] = df['Posts'].apply(get_rhyme_scheme)
    
    # sentiment analysis
    df['sentiment_polarity'] = df['Posts'].apply(get_vader_scores)

    df.to_csv("data/feature_enriched_posts.tsv", sep = '\t', index = False)
    