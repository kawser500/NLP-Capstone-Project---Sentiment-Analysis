import spacy 
from textblob import TextBlob
import pandas as pd

nlp = spacy.load("en_core_web_sm")

# Function to clean text and remove stop words
def clean_text(text):
    doc = nlp(text)
    
    # Remove stop words and non-alphabetic tokens, and lemmatize the remaining tokens
    cleaned_text = " ".join(token.lemma_ for token in doc if not token.is_stop and token.is_alpha)
    
    return cleaned_text

# Function for sentiment analysis using TextBlob
def analyze_sentiment_textblob(text):
    blob = TextBlob(text)
    
    # Get the polarity score (range -1 to 1)
    polarity_score = blob.sentiment.polarity
    
    # Classify the sentiment based on the polarity score
    if polarity_score > 0:
        sentiment = 'positive'
    elif polarity_score == 0:
        sentiment = 'neutral'
    else:
        sentiment = 'negative'
    
    return sentiment


# This is the file path on my personal machine to access the CSV file. 
# Generic read_csv used below for submission purposes. 
# file_path = r"\Users\messi\Documents\Bootcamp - Python\Tasks\Task 21\amazon_product_reviews.csv"
# df = pd.read_csv(file_path, dtype={1: str, 10: str})



try:
    print("Attempting to load CSV file...")
    df = pd.read_csv("amazon_product_reviews.csv", dtype={1: str, 10: str})
    print("CSV file loaded successfully.")

    # Sample a small portion of the data
    sample_size = 100  
    df_sample = df.sample(n=sample_size, random_state=1)  
    
    # Drop rows with missing values in the "reviews.text" column
    df_cleaned = df_sample.dropna(subset=["reviews.text"])

    # Clean the text in the "reviews.text" column
    print("Cleaning text...")
    df_cleaned["cleaned_reviews"] = df_cleaned["reviews.text"].apply(clean_text)
    print("Text cleaning completed.")

    print("Performing sentiment analysis...")

    # Apply sentiment analysis using TextBlob to the cleaned reviews
    df_cleaned["sentiment_textblob"] = df_cleaned["cleaned_reviews"].apply(analyze_sentiment_textblob)

    # Count the number of positive and negative reviews for both TextBlob and transformer-based model
    sentiment_counts_textblob = df_cleaned["sentiment_textblob"].value_counts()

    print("Sentiment analysis completed.\n")

    print("Cleaned Reviews:")
    print(df_cleaned["cleaned_reviews"])
    print("\nSentiment Analysis (TextBlob):")
    print(df_cleaned["sentiment_textblob"])
    print("\nSentiment Counts (TextBlob):")
    print(sentiment_counts_textblob)
    
except Exception as e:
    print("An error occurred:", e)



"""
Note: alternative way to perform sentiment analysis (more accuracy) would be to use a spaCy transformer 
This is not included in the code above as it is too computationally expensive
Code included below:

Function for sentiment analysis using spaCy transformer-based model
def analyze_sentiment_transformer(text):
    nlp = spacy.load("en_core_web_trf")
    doc = nlp(text)
    
    # Extract sentiment score from the document
    sentiment_score = doc.sentiment
    if sentiment_score >= 0.5:
        sentiment = 'positive'
    elif sentiment_score <= -0.5:
        sentiment = 'negative'
    else:
        sentiment = 'neutral'
    
    return sentiment

  Apply sentiment analysis using spaCy transformer-based model to the cleaned reviews
    df_cleaned["sentiment_transformer"] = df_cleaned["cleaned_reviews"].apply(analyze_sentiment_transformer)

 sentiment_counts_transformer = df_cleaned["sentiment_transformer"].value_counts()

print("\nSentiment Analysis (spaCy):")
print(df_cleaned["sentiment_transformer"])
print("\nSentiment Counts (spaCy):")
print(sentiment_counts_transformer)

"""