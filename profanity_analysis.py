from better_profanity import profanity
import spacy

nlp = spacy.load("en_core_web_sm")

# Remove cuss words from a song.
def get_sentences(text):
    doc = nlp(text)
    return [sent.text for sent in doc.sents]

def clean_profanity_from_sentences(sentences):
    cleaned_sentences = []
    for sentence in sentences:
        cleaned_sentence = profanity.censor(sentence)
        cleaned_sentences.append(cleaned_sentence)
    return cleaned_sentences

def clean_document(text):
    sentences = get_sentences(text)
    cleaned_sentences = clean_profanity_from_sentences(sentences)
    clean_text = ' '.join(cleaned_sentences)
    return clean_text

# Count how many cuss words there are in a document
def count_profanity_in_a_sentence(sentence):
    count = 0
    words = sentence.split()
    for word in words:
        if (profanity.contains_profanity(word)):
            count += 1
    return count

def count_profanity(text):
    total_count = 0
    sentences = get_sentences(text)
    for sentence in sentences:
        total_count += count_profanity_in_a_sentence(sentence)
    return total_count