# Text Stylistic Features Extraction

## Description

This stylistic feature extraction method analyzes the way text is written rather than what is written. It identifies linguistic and structural patterns across multiple categories, e.g., lexical features (tokens, type-token-ratio, root-type-token-ratio, corrected-type-token-ratio, herdan, and maas), profanity features (profanity word count, text without profanity),  readability (flesch_reading_ease, text_standard, reading_time, syllable_count, sentence_count, char_count, polysyllable_count, get_monosyllable_count), # repetitiveness (num_of_choruses), rhyme (rhyme_scheme), and sentiment analysis (sentiment_polarity). These features capture the author's writing style, tone and expressiveness. By quantifying the stylistic features, the method provides insights into the underlying communication style beyong the content itself.

## Use Cases

- The content writters are needed to be grouped based on their writing skills using stylistic features of their written samples. 
- Educational and editorial systems want to evaluate whether a piece of writing match the desired readability and tone criteria e.g., formal academic writings or simplifying public-face documents.

## Input Data

The input data consists of social media posts (one per line) as a CSV file, i.e., `data/input_social_posts.csv`. The following are a few examples:

| posts |
|---------|
| The moon doesn’t answer, but it listens better than most. | 
| I left my thoughts in yesterday’s rain. |
| Some silences speak louder than storms. |
| The stars look different when you’re missing someone. |
| ... |
## Output Data

The method writes output to a CSV file, i.e., `data/output_posts_with_entities.csv`. It has the first column as the original post's text, followed by columns representing entities extracted from the text. Each column value is a list of one or more entities extracted from a post.

| Posts | tokens | ttr | rttr | cttr | herdan | maas | profanity_count |	profanity_cleaned_text |	flesch_reading_ease |	text_standard |	reading_time |	syllable_count |	sentence_count |	char_count |	polysyllable_count |	get_monosyllable_count |	num_of_choruses |	rhyme_scheme |	sentiment_polarity |
|----|-----|------|-----|-----|------|-------|-------|---------|----------|------------|----------|--------|-------|--------|-------|--------|--------|----------|-----------|
| The moon doesn’t answer, but it listens better than most.	| 10	| 1.0	| 3.162277660168379	2.23606797749979	1.0	0.0	0	The moon doesn’t answer, but it listens better than most.	78.24500000000002	3rd and 4th grade	0.70512	14	1	48	0	6	0	A	positive
I left my thoughts in yesterday’s rain.	7	1.0	2.6457513110645903	1.8708286933869707	1.0	0.0	0	I left my thoughts in yesterday’s rain.	90.95857142857145	8th and 9th grade	0.48477	9	1	33	1	6	0	A	neutral
Some silences speak louder than storms.	6	1.0	2.4494897427831783	1.7320508075688774	1.0	0.0	0	Some silences speak louder than storms.	73.84500000000001	8th and 9th grade	0.49945999999999996	9	1	34	1	4	0	A	neutral
The stars look different when you’re missing someone.	8	1.0	2.82842712474619	2.0	1.0	0.0	0	The stars look different when you’re missing someone.	71.81500000000001	8th and 9th grade	0.67574	12	1	46	1	5	0	A	negative
|...|


## Hardware Requirements

The method runs on a small virtual machine provided by a cloud computing company (2 x86 CPU cores, 4 GB RAM, 40 GB HDD).

## Environment Setup

use the following envrionment to deploy the virtual working environment. It deploys Python=3.10 as part of the dependencies.

  ```conda env create -f environment.yml```

## How to Use

- The script ```main.py``` calls all the feature extraction functions defined ```features_util/```. It reads the input data ```data/input_posts.tsv``` and writes the output as the original posts with stylistic features in separate columns in file ```stylistic_features_enriched_posts.tsv```. You can execute the script using the following command from the project main directory.

- ```paython main.py```  

- Populate the input file `data/input_posts.tsv` with social media posts on the topic of interest, keeping one per row (Optional: the file already has sample posts). 

## Technical Details

For assessing **lexical richness**, we employ the **LexicalRichness** package ([https://lexicalrichness.readthedocs.io/en/latest/](https://lexicalrichness.readthedocs.io/en/latest/)), which provides quantitative measures of vocabulary diversity in a text, such as the Type–Token Ratio (TTR), Root TTR, herdan, and maas. These metrics capture how varied and sophisticated the vocabulary is, helping evaluate writing quality, vocabulary knowledge, or linguistic competence. For **readability features**, we use **TextStat** ([https://pypi.org/project/textstat/](https://pypi.org/project/textstat/)), which computes indices like the Flesch Reading Ease and reading time. For **sentiment analysis**, we rely on **VADER Sentiment** ([https://pypi.org/project/vaderSentiment/](https://pypi.org/project/vaderSentiment/)). For **profanity detection**, we integrate **Better Profanity** ([https://pypi.org/project/better-profanity/](https://pypi.org/project/better-profanity/)), a lightweight and fast Python library for identifying and censoring offensive language. While the rhyme features are identified through regular expressions.

The method builds on Omar Chouikha's original work for his thesis [https://github.com/bolandka/writing-style-lyrics](https://github.com/bolandka/writing-style-lyrics).

## Contact
For more information, please contact <katarina.boland@hhu.de>



