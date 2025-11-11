# Text Stylistic Features Extraction

## Description

This stylistic feature extraction method analyzes the way text is written rather than what is written. It identifies linguistic and structural patterns across multiple categories, e.g., lexical features (tokens, ttr, rttr, cttr, herdan, and maas), profanity features (profanity word count, text without profanity),  readability (flesch_reading_ease, text_standard, reading_time, syllable_count, sentence_count, char_count, polysyllable_count, get_monosyllable_count), # repetitiveness (num_of_choruses), rhyme (rhyme_scheme), and sentiment analysis (sentiment_polarity). These features capture the author's writing style, tone and expressiveness. By quantifying the stylistic features, the method provides insights into the underlying communication style beyong the content itself.

## Use Cases

- The content writters are needed to be grouped based on their writing skills using stylistic features of their written samples. 
- Educational and editorial systems want to evaluate whether a piece of writing match the desired readability and tone criteria e.g., formal academic writings or simplifying public-face documents.

## Input Data

The input data consists of social media posts (one per line) as a CSV file, i.e., `data/input_social_posts.csv`. The following are a few examples:

|Posts|
|---------|
|"@bob@infosec.exchange #Crypto €BMW ""Let’s go!"" https://t.co/xyz123 😀"|
|#Startups 💡 $GOOG https://t.co/xyz123 @dave@mastodon.social 'Not sure about this'|
|@bob@mastodon.social $AAPL 'This is amazing' 😀 #Crypto https://news.site/article|
|"@dave@infosec.exchange ""Exciting times ahead!"" https://t.co/xyz123 €BMW #AI 😀"|
|#AI @bob@mastodon.social €ETH 🚀 'Not sure about this' https://news.site/article|
|...|

## Output Data

The method writes output to a CSV file, i.e., `data/output_posts_with_entities.csv`. It has the first column as the original post's text, followed by columns representing entities extracted from the text. Each column value is a list of one or more entities extracted from a post.

| Posts | tokens | ttr | rttr | cttr | herdan | maas | profanity_count |	profanity_cleaned_text |	flesch_reading_ease |	text_standard |	reading_time |	syllable_count |	sentence_count |	char_count |	polysyllable_count |	get_monosyllable_count |	num_of_choruses |	rhyme_scheme |	sentiment_polarity |
|----|-----|------|-----|-----|------|-------|-------|---------|----------|------------|----------|--------|-------|--------|-------|--------|--------|----------|-----------|
| @bob@infosec.exchange #Crypto €BMW "Let’s go!" https://t.co/xyz123 😀 |	12 |	1.0	| 3.464101615137755 |	2.4494897427831783 |	1.0 |	0.0 |	0 |	@bob@infosec.exchange #Crypto €BMW "Let’s go!" https://t.co/xyz123 😀	| 17.44500000000002	| 11th and 12th grade	| 0.9107799999999999	| 13	| 1	| 62	| 2	| 3	| 0	| A	| positive |
| #Startups 💡 $GOOG https://t.co/xyz123 @dave@mastodon.social 'Not sure about this'	| 14	| 1.0	| 3.7416573867739413	| 2.6457513110645903	| 1.0	| 0.0	| 0	| #Startups 💡 $GOOG https://t.co/xyz123 @dave@mastodon.social 'Not sure about this' |	44.150000000000006	| 11th and 12th grade	| 1.0723699999999998	| 15	| 2	| 73	| 2	| 4	| 0	|A	| negative |
| @bob@mastodon.social $AAPL 'This is amazing' 😀 #Crypto https://news.site/article	| 13	| 1.0	| 3.6055512754639896	| 2.5495097567963927	| 1.0	| 0.0	| 0	| @bob@mastodon.social $AAPL 'This is amazing' 😀 #Crypto https://news.site/article |	18.44428571428574	| 12th and 13th grade	| 1.0723699999999998	| 15	| 1	| 73	| 3	| 3	| 0	| A	| positive |
|...|


## Hardware Requirements

The method runs on a small virtual machine provided by a cloud computing company (2 x86 CPU cores, 4 GB RAM, 40 GB HDD).

## Environment Setup

The method is tested with Python 3.10 and should work with other Python versions as well. Use the following command to setup the virtual working environment by installing all dependencies;

  ```pip install -r requirements.txt```

## How to Use

- Open `index.ipynb` and execute the cells to use the method. It imports and uses the entity extraction function defined in `entity_extractor.py`.
- Populate the input file `data/input_social_posts.csv` with social media posts on the topic of interest, keeping one per row (Optional: the file already has sample posts). 

## Technical Details

## References

## Contact
