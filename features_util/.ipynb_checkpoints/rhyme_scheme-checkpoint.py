import re
import string
import nltk
import math
from collections import defaultdict
from collections import Counter
from nltk.corpus import cmudict
from nltk.tokenize import word_tokenize

nltk.download('cmudict')
nltk.download('punkt_tab')

# Load CMU Pronouncing Dictionary
d = cmudict.dict()

# Extracts the last word from a given line.
def get_last_word(line):
    tokens = word_tokenize(line)
    return tokens[-1].lower() if tokens else ''

# Removes the punctuation if there is any.
def remove_trailing_punctuation(sentence):
    return sentence.rstrip(string.punctuation)

# Get the rhyme scheme of a section.
def get_rhyme_scheme(text):
    lines = text.strip().split('\n')
    rhyme_dict = defaultdict(lambda: len(rhyme_dict))
    scheme = []
    
    for line in lines:
        line = remove_trailing_punctuation(line)
        last_word = get_last_word(line)
        if last_word:
            # Get the phonemes of the last word
            phonemes = d.get(last_word, [last_word])[0]
            # Create a simplified rhyme key from the phonemes
            rhyme_key = phonemes[-1] # Simplified approach for rhyme
            scheme.append(rhyme_dict[rhyme_key])
            
    return ', '.join(chr(ord('A') + r) for r in scheme)

# Handles the duplicate sections so that when there is a new Chorus that has a different rhyme scheme, it doesn't overwrite the old one
# But instead, the section name would be Chorus 2.
def handle_duplicate_sections(matches):
    # Dictionary to maintain count of section occurrences
    section_count = defaultdict(int)
    
    # Result dictionary with section name and text
    formatted_dict = {}
    for section, text in matches:

        section = section.strip()
        text = text.strip()
        section_count[section] += 1
    
        # Create a unique key for each section occurrence
        section_key = f"{section} {section_count[section]}"
    
        if section_key not in formatted_dict:
            formatted_dict[section_key] = []
    
        formatted_dict[section_key].append(text)

    # Format the output
    final_dict = {key: "\n".join(texts) for key, texts in formatted_dict.items()}

    return final_dict

# Get the rhyme schemes of a song by section, e.g.: Verse 1: 'A,B,A,C' Chorus 1: 'A,B,A,B'.
def get_rhyme_scheme_by_section(lyrics):
    pattern = r'\[(.*?)\]\s*([^[]*)'

    # Find all matches in the lyrics text
    matches = re.findall(pattern, lyrics)
    sections = handle_duplicate_sections(matches)
    

    rhyme_schemes = {}
    for section, text in sections.items():
        # Determine the name of the section
        rhyme_scheme = get_rhyme_scheme(text)
        rhyme_schemes[section] = rhyme_scheme
    
    return rhyme_schemes

# Get the number of different rhyme schemes throughout the song.
def get_different_rhyme_schemes(rhyme_schemes):

    # Set to store unique patterns
    unique_patterns = set()

    for section in rhyme_schemes.values():
        unique_patterns.add(section)

    return len(unique_patterns)

def get_unique_rhyme_labels(rhyme_schemes):
    unique_rhyme_labels = set()

    for scheme in rhyme_schemes.values():
        labels = scheme.split(', ')
        unique_rhyme_labels.update(labels)

    return len(unique_rhyme_labels)   

def get_rhyme_repetition_rate(rhyme_schemes):
    total_labels = 0
    total_repetition_rate = 0

    for scheme in rhyme_schemes.values():
        labels = scheme.split(', ')
        label_counts = Counter(labels)
        
        # Update overall counts
        total_labels += len(labels)        
        
        # Section Repetition Rate
        repetition_rate = sum(count - 1 for count in label_counts.values()) / len(labels)
        total_repetition_rate += repetition_rate

    return total_repetition_rate / total_labels

def get_rhyme_entropy(rhyme_schemes):
    total_labels = 0
    total_entropy = 0

    for scheme in rhyme_schemes.values():
        labels = scheme.split(', ')
        label_counts = Counter(labels)
        
        # Update overall counts
        total_labels += len(labels)        
        
        # Section Entropy
        probabilities = [count / len(labels) for count in label_counts.values()]
        entropy = -sum(p * math.log(p, 2) for p in probabilities)
        total_entropy += entropy * len(labels)  # Weighted by section length

    return total_entropy / total_labels

def get_average_rhyme_length(rhyme_schemes):
    total_labels = 0
    section_count = len(rhyme_schemes)

    for scheme in rhyme_schemes.values():
        labels = scheme.split(', ')
        total_labels += len(labels)

    return total_labels / section_count

# Single function to process all metrics at once
def process_rhyme_metrics(lyrics):
    rhyme_schemes = get_rhyme_scheme_by_section(lyrics)
    return {
        'different_rhyme_schemes': get_different_rhyme_schemes(rhyme_schemes),
        'unique_rhyme_labels': get_unique_rhyme_labels(rhyme_schemes),
        'rhyme_repetition_rate': get_rhyme_repetition_rate(rhyme_schemes),
        'rhyme_entropy': get_rhyme_entropy(rhyme_schemes),
        'average_rhyme_length': get_average_rhyme_length(rhyme_schemes),
    }