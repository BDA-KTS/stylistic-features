from transformers import pipeline
import re
import torch
import pandas as pd

class PunctuationModel():
    def __init__(self, model='oliverguhr/fullstop-punctuation-multilang-large') -> None:
        # Check if a GPU is available
        if torch.cuda.is_available():
            self.pipe = pipeline("ner", model=model, grouped_entities=False, device=0)
        else:
            self.pipe = pipeline("ner", model=model, grouped_entities=False)        

    def preprocess(self,text):
        #remove markers except for markers in numbers 
        text = re.sub(r"(?<!\d)[.,;:!?](?!\d)","",text)
        text = re.sub(r'[^\x00-\x7F]', '', text)
        #todo: match acronyms https://stackoverflow.com/questions/35076016/regex-to-match-acronyms
        text = text.split()
        return text

    def restore_punctuation(self,text):        
        result = self.predict(self.preprocess(text))
        return self.prediction_to_text(result)
        
    def overlap_chunks(self,lst, n, stride=0):
        """Yield successive n-sized chunks from lst with stride length of overlap."""
        for i in range(0, len(lst), n-stride):
                yield lst[i:i + n]

    def predict(self,words):
        overlap = 5
        chunk_size = 230
        if len(words) <= chunk_size:
            overlap = 0

        batches = list(self.overlap_chunks(words,chunk_size,overlap))

        # if the last batch is smaller than the overlap, 
        # we can just remove it
        if len(batches[-1]) <= overlap:
            batches.pop()

        tagged_words = []     
        for batch in batches:
            # use last batch completely
            if batch == batches[-1]: 
                overlap = 0
            text = " ".join(batch)
            result = self.pipe(text)      
            print(f"Text length: {len(text)}, Result end: {result[-1]['end']}")
            

            assert len(text) == result[-1]["end"], text+" chunk size too large, text got clipped"
                
            char_index = 0
            result_index = 0
            for word in batch[:len(batch)-overlap]:                
                char_index += len(word) + 1
                # if any subtoken of an word is labled as sentence end
                # we label the whole word as sentence end        
                label = 0
                while result_index < len(result) and char_index > result[result_index]["end"] :
                    label = result[result_index]['entity']
                    score = result[result_index]['score']
                    result_index += 1                        
                tagged_words.append([word,label, score])
        
        assert len(tagged_words) == len(words)
        return tagged_words

    def prediction_to_text(self,prediction):
        result = ""
        for word, label, _ in prediction:
            result += word
            if label == "0":
                result += " "
            if str(label) in ".,?-:":
                result += label+" "
        return result.strip()
    
df = pd.read_csv('datasets/clean_dataset.csv')    

punctuation_model = PunctuationModel(model="oliverguhr/fullstop-punctuation-multilang-large")

def restore_punctuation(text):
    return punctuation_model.restore_punctuation(text)

def restore_punctuation_safe(text):
    try:
        return punctuation_model.restore_punctuation(text)
    except Exception as e:
        print(f"Error occurred for text: {text[:30]}... - Skipping. Error: {e}")
        return None  # Return None or an empty string to skip this song    
    
df['punctuated_text'] = df['clean_lyrics'].map(restore_punctuation_safe)
df.to_csv('datasets/preprocessed_dataset.csv', index=False)