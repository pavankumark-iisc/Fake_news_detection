import pandas as pd
import re
import unicodedata

#loading csv
df = pd.read_csv("news_dataset.csv")  #downloaded from: https://www.kaggle.com/datasets/imbikramsaha/fake-real-news?resource=download

#using sentence splitter regex
sentence_splitter = re.compile(r'(?<=[.!?]) +')

#cleaning and truncate function
def clean_text_final(text):
    text = str(text)
    
    #normalizing and removing invisible characters
    text = unicodedata.normalize("NFKD", text)

    #removing non-ASCII
    text = text.encode("ascii", errors="ignore").decode()

    #removing quotes, slashes, hashtags, etc.
    text = re.sub(r'[\'"\\/#]', '', text)

    #replacing newlines, tabs with space
    text = re.sub(r'[\n\r\t]', ' ', text)

    #normalizing whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    #split into sentences and keep only the first 3 lines
    sentences = sentence_splitter.split(text)
    text = ' '.join(sentences[:3])

    return text

#apply cleaning
df['text'] = df['text'].apply(clean_text_final)

#saving the cleaned file into a new .csv file
df.to_csv("final_short_clean_news_dataset.csv", index=False)

