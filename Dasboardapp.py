import streamlit as st
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from textblob import TextBlob
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

st.title("Summarized Data Visualization & Reporting Dashboard")

uploaded_file = st.file_uploader("Upload your summarized CSV", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Data Preview", df.head())

    st.header("Word Cloud (Key Themes)")
    text = ' '.join(df['generated_summary'].astype(str))
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig1, ax1 = plt.subplots()
    ax1.imshow(wordcloud, interpolation='bilinear')
    ax1.axis('off')
    st.pyplot(fig1)

    st.header("Sentiment Distribution")
    df['sentiment'] = df['generated_summary'].astype(str).apply(lambda x: TextBlob(x).sentiment.polarity)
    df['sentiment_label'] = pd.cut(df['sentiment'], bins=[-1, -0.1, 0.1, 1], labels=['Negative', 'Neutral', 'Positive'])
    sent_counts = df['sentiment_label'].value_counts().reindex(['Positive', 'Neutral', 'Negative']).fillna(0)
    fig2, ax2 = plt.subplots()
    sent_counts.plot(kind='bar', color=['green', 'gray', 'red'], ax=ax2)
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Frequency')
    st.pyplot(fig2)

    st.header("Topic Distribution")
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    X = vectorizer.fit_transform(df['generated_summary'].astype(str))
    lda = LatentDirichletAllocation(n_components=5, random_state=42)
    lda_topics = lda.fit_transform(X)
    topic_counts = lda_topics.argmax(axis=1)
    topic_distribution = pd.Series(topic_counts).value_counts().sort_index()
    fig3, ax3 = plt.subplots()
    topic_distribution.plot(kind='bar', ax=ax3)
    plt.title('Topic Distribution')
    plt.xlabel('Topic')
    plt.ylabel('Article Count')
    st.pyplot(fig3)

    st.header("Summary & Recommendations")
    st.markdown(f"""
        - **Total summaries analyzed:** {len(df)}
        - **Most common sentiment:** {sent_counts.idxmax()}
        - **Number of topics discovered:** {len(topic_distribution)}
        - **Actionable Insights:**  
          Review topics with highest counts for dominant themes. Investigate outlier sentiment articles for further analysis.
    """)
