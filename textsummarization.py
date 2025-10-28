import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
from transformers import pipeline

nltk.download('punkt')

def extractive_summary(text, n_sentences=3):
    sentences = sent_tokenize(text)
    return ' '.join(sentences[:min(n_sentences, len(sentences))])

summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)

def abstractive_summary(text, max_length=130, min_length=30):
    input_text = text if len(text) < 1024 else text[:1024]
    result = summarizer(input_text, max_length=max_length, min_length=min_length, do_sample=False)
    return result[0]['summary_text']

def summarize_csv(csv_path, text_column='article', extractive=True):
    df = pd.read_csv(csv_path)
    summaries = []
    for idx, row in df.iterrows():
        text = row[text_column]
        if extractive:
            summary = extractive_summary(text)
        else:
            summary = abstractive_summary(text)
        summaries.append(summary)
        if (idx+1)%100 == 0:
            print(f'Summarized {idx+1} rows')
    df['generated_summary'] = summaries
    output_path = csv_path.replace('.csv', '_summarized.csv')
    df.to_csv(output_path, index=False)
    print(f'Saved summarized CSV: {output_path}')

test_path = r'C:\Users\Varun Boga\OneDrive\Desktop\Infosys\Text Summarization\CNN_data\cnn_dailymail_test.csv'
valid_path = r'C:\Users\Varun Boga\OneDrive\Desktop\Infosys\Text Summarization\CNN_data\cnn_dailymail_valid.csv'

summarize_csv(test_path, text_column='article', extractive=True)
summarize_csv(valid_path, text_column='article', extractive=False)
