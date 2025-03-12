from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import os
import joblib
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer


# Parent class
class BertopicModeling:

    def __init__(self, clean_lyrics, topic_model_path=None):
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        # Remove the frequent words
        ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
        # Remove the stop words
        vectorizer_model = CountVectorizer(stop_words="english")

        cluster_model = KMeans(n_clusters=50, random_state=42)
        
        # Load the model if a path is provided; otherwise, initialize a new model
        if topic_model_path:
            self.topic_model = BERTopic.load(topic_model_path)
        else:
            self.topic_model = BERTopic(
                hdbscan_model=cluster_model,
                vectorizer_model=vectorizer_model,
                embedding_model=self.embedding_model,
                ctfidf_model=ctfidf_model
            )
        self.topics = None
        self.probs = None

    def save_model(self, save_path):
        # Create a directory to save the models if it doesn't exist
        os.makedirs(save_path, exist_ok=True)

        # Save the BERTopic model using joblib
        joblib.dump(self.topic_model, os.path.join(save_path, 'bertopic_model.pkl'), compress=3)

    def get_topics(self):
        return self.topic_model.get_topic_info()

    def get_top(self):
        return self.topics
    
    def get_document(self):
        return self.topic_model.get_document_info()

# Subclass for training
class BertopicTraining(BertopicModeling):
    
    def __init__(self, clean_lyrics):
        super().__init__(clean_lyrics)
        self.topics, self.probs = self.topic_model.fit_transform(clean_lyrics)  # Use fit_transform for training

    def save(self, save_path):
        self.save_model(save_path)    

# Subclass for testing
class BertopicTesting(BertopicModeling):
    
    def __init__(self, clean_lyrics, trained_model_path):
        super().__init__(clean_lyrics, topic_model_path=trained_model_path)
        self.topics, self.probs = self.topic_model.transform(clean_lyrics)  # Use transform for testing
