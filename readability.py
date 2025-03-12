import textstat

#Returns the Flesch Reading Ease Score. 
#While the maximum score is 121.22, there is no limit on how low the score can be. A negative score is valid.
def get_flesch_reading_ease(text):
    return textstat.flesch_reading_ease(text)

#Based upon all the possible tests, returns the estimated school grade level required to understand the text.
def get_text_standard(text):
    return textstat.text_standard(text, float_output=False)

#Returns the reading time of the given text.
#Assumes 14.69ms per character.
def get_reading_time(text):
    return textstat.reading_time(text, ms_per_char=14.69)

#Returns the number of syllables present in the given text.
def get_syllable_count(text):
    return textstat.syllable_count(text)

#Returns the number of sentences present in the given text.
def get_sentence_count(text):
    return textstat.sentence_count(text)

#Returns the number of characters present in the given text.
def get_char_count(text):
    return textstat.char_count(text, ignore_spaces=True)

#Returns the number of words with a syllable count greater than or equal to 3.
def get_polysyllable_count(text):
    return textstat.polysyllabcount(text)

#Returns the number of words with a syllable count equal to one.
def get_monosyllable_count(text):
    return textstat.monosyllabcount(text)