import sys
sys.path.append(r"C:\Users\Varun Boga\OneDrive\Desktop\AI Narrative Nexus\scr\Topic_Modeling")
import text_processing # type: ignore

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
import os
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns

def load_topic_models():
    lda = joblib.load('model/Topic_modeling/lda_model.pkl')
    vectorizer = joblib.load('model/Topic_modeling/lda_vectorizer.pkl')
    return lda, vectorizer

def load_sentiment_models():
    rf = joblib.load('model/Sentiment_analysis/random_forest_model.pkl')
    vec = joblib.load('model/Sentiment_analysis/tfidf_vectorizer.pkl')
    return rf, vec

def predict_topics(texts, lda, vectorizer):
    X_counts = vectorizer.transform(texts)
    doc_topics = lda.transform(X_counts)
    assigned_topics = np.argmax(doc_topics, axis=1)
    return assigned_topics

def predict_sentiments(texts, rf, vec):
    X = vec.transform(texts)
    preds = rf.predict(X)
    return preds

def analyze_data(df):
    lda, lda_vec = load_topic_models()
    rf, rf_vec = load_sentiment_models()
    if "text" not in df.columns:
        raise ValueError("DataFrame must have a 'text' column")
    texts = df["text"].astype(str).tolist()
    topic_labels = predict_topics(texts, lda, lda_vec)
    sentiment_labels = predict_sentiments(texts, rf, rf_vec)
    df["assigned_topic"] = topic_labels
    df["sentiment_label"] = sentiment_labels
    return df

st.set_page_config(page_title="AI Narrative Nexus", layout="wide")

st.sidebar.title("AI Narrative Nexus")
page = st.sidebar.radio("Navigation", [
    "Home", 
    "Upload & Analyze", 
    "Topic Modeling", 
    "Sentiment Analysis", 
    "Summarization", 
    "Data Visualization", 
    "Live Demo", 
    "About"
])

st.sidebar.markdown("**Select Model**")
model_option = st.sidebar.selectbox(
    "Active Model",
    ["TopicModel-LDA", "SentimentNet", "Summarizer-BART"]
)
st.sidebar.success(f"Using: {model_option} (active)", icon="✅")

if page == "Home":
    st.markdown(
        "<h1 style='text-align:center;margin-bottom:1.5rem'>🧠 AI Narrative Nexus</h1>", unsafe_allow_html=True
    )
    st.markdown(
        "<h3 style='text-align:center;margin-bottom:2rem'>Unlocking Insights from News & Text with AI!</h3>",
        unsafe_allow_html=True
    )
    with st.expander("📑 Project Overview", expanded=True):
        st.markdown("""
AI Narrative Nexus is an interactive platform that transforms raw news/text into actionable insights using modern Natural Language Processing:

- **Topic Modeling:** Discover dominant topics in news and text datasets.
- **Sentiment Analysis:** Gauge the emotional tone or polarity at scale.
- **Intelligent Summarization:** Generate concise, relevant summaries.
- **Visualization Suite:** Quickly explore trends, clusters, and outliers.

Whether you're a researcher, journalist, data scientist, or innovator — our tool empowers rapid, scalable text exploration and understanding.
        """)

    st.markdown("---")
    with st.expander("✨ Key Features (click to expand)"):
        st.markdown("""
- 🔍 **Upload & Analyze:** Instantly process your own datasets—just upload and go!
- 📊 **Interactive Dashboards:** Multiple dynamic charts for every NLP module.
- 🧩 **Model Flexibility:** Switch between topic, sentiment, and summarization models.
- 🚀 **Fast Summarization:** Uses state-of-the-art transformer technology.
- 📈 **Data Exports:** Download annotated results for sharing and reporting.
        """)

    st.markdown("---")
    with st.expander("🌐 Try a Demo Workflow!"):
        st.markdown("""
1. **Upload** a CSV/text file or paste a news article.
2. **Choose** your analysis: Topic, Sentiment, or Summarize.
3. **Visualize** the patterns in real time.
Note: Demo data and model options are preloaded for a fast start!
        """)
    st.markdown("---")
    st.info("Built as part of the Infosys Internship 2025 by Your Name. Full documentation and report available in the 'About' section.")
    st.markdown(
        "<div style='color:#888'>Questions? Feedback? Connect via email or check out the About page for more.</div>",
        unsafe_allow_html=True
    )

elif page == "Upload & Analyze":
    from transformers import pipeline
    st.header("Upload & Analyze")
    uploaded_file = st.file_uploader("Upload a CSV file with a column to analyze", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if "text" not in df.columns:
            if "article" in df.columns:
                df = df.rename(columns={"article": "text"})
            elif "generated_summary" in df.columns:
                df = df.rename(columns={"generated_summary": "text"})
            elif "highlights" in df.columns:
                df = df.rename(columns={"highlights": "text"})
        if "text" not in df.columns:
            st.error("CSV must contain a column named 'text', 'article', 'generated_summary', or 'highlights'.")
        else:
            st.success(f"File uploaded: {uploaded_file.name}")
            st.write(df.head())
            st.subheader("Summaries (First 5 Rows)")
            summarizer = pipeline("summarization")
            for i, row in df.head(5).iterrows():
                text = str(row["text"])
                if text and text.strip():
                    try:
                        summary_results = summarizer(text, max_length=80, min_length=30, do_sample=False)
                        if summary_results and isinstance(summary_results, list) and 'summary_text' in summary_results[0]:
                            summary = summary_results[0]['summary_text']
                            st.markdown(f"**Row {i+1}:** {summary}")
                        else:
                            st.markdown(f"**Row {i+1}:** [No summary available]")
                    except Exception as e:
                        st.markdown(f"**Row {i+1}:** Summary failed: {e}")
                else:
                    st.markdown(f"**Row {i+1}:** [Empty or blank row]")

            st.subheader("Topic Modeling (First 5 Rows)")
            lda_path = "model/Topic_modeling/topic_classifier.pkl"
            vect_path = "model/Sentiment_analysis/tfidf_vectorizer.pkl"
            if os.path.exists(lda_path) and os.path.exists(vect_path):
                lda = joblib.load(lda_path)
                vectorizer = joblib.load(vect_path)
                X_counts = vectorizer.transform(df["text"].astype(str).head(5))
                topics = lda.predict(X_counts)
                for idx, topic in enumerate(topics):
                    st.markdown(f"**Row {idx+1}: Assigned Topic:** {topic}")
                if "true_topic" in df.columns:
                    y_true = df["true_topic"].head(5)
                    y_pred = topics
                    acc = accuracy_score(y_true, y_pred)
                    cm = confusion_matrix(y_true, y_pred)
                    report = classification_report(y_true, y_pred, output_dict=True)

                    st.subheader("Topic Model Accuracy")
                    st.write(f"Accuracy: {acc:.2f}")

                    st.subheader("Confusion Matrix")
                    fig_cm, ax_cm = plt.subplots()
                    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=ax_cm)
                    st.pyplot(fig_cm)

                    st.subheader("Classification Report")
                    st.dataframe(pd.DataFrame(report).transpose())
            else:
                st.error("Topic model or vectorizer file not found.")

elif page == "Topic Modeling":
    st.header("Topic Modeling - Visualization Gallery")
    df = pd.read_csv("data/processed/cnn_dailymail_test_summarized.csv")
    if "text" not in df.columns:
        if "article" in df.columns:
            df = df.rename(columns={"article": "text"})
        elif "generated_summary" in df.columns:
            df = df.rename(columns={"generated_summary": "text"})
        elif "highlights" in df.columns:
            df = df.rename(columns={"highlights": "text"})
    df = analyze_data(df)
    topic_labels, topic_counts = np.unique(df["assigned_topic"], return_counts=True)
    topic_names = [str(t) for t in topic_labels]
    counts = topic_counts.tolist()
    data = pd.DataFrame({"Topic": topic_names, "Count": counts})
    chart_type = st.selectbox(
        "Choose Chart Type",
        [
            "Bar Chart",
            "Pie Chart",
            "Horizontal Bar Chart",
            "Line Chart",
            "Area Chart",
            "Scatter Plot",
            "Donut Chart",
            "Box Plot"
        ]
    )
    if chart_type == "Bar Chart":
        fig, ax = plt.subplots()
        ax.bar(data["Topic"], data["Count"], color="skyblue")
        ax.set_ylabel("Count")
        st.pyplot(fig)
    elif chart_type == "Pie Chart":
        fig, ax = plt.subplots()
        ax.pie(data["Count"], labels=data["Topic"], autopct='%1.1f%%')
        st.pyplot(fig)
    elif chart_type == "Horizontal Bar Chart":
        fig, ax = plt.subplots()
        ax.barh(data["Topic"], data["Count"], color="coral")
        ax.set_xlabel("Count")
        st.pyplot(fig)
    elif chart_type == "Line Chart":
        fig, ax = plt.subplots()
        ax.plot(data["Topic"], data["Count"], marker='o', linestyle='-', color='green')
        ax.set_ylabel("Count")
        st.pyplot(fig)
    elif chart_type == "Area Chart":
        st.area_chart(data.set_index("Topic"))
    elif chart_type == "Scatter Plot":
        fig, ax = plt.subplots()
        ax.scatter(data["Topic"], data["Count"], s=200, color="purple")
        ax.set_ylabel("Count")
        st.pyplot(fig)
    elif chart_type == "Donut Chart":
        fig, ax = plt.subplots()
        wedges, texts, autotexts = ax.pie(data["Count"], labels=data["Topic"], autopct='%1.1f%%', wedgeprops={"width":0.5})
        plt.setp(autotexts, size=10, weight="bold")
        ax.set(aspect="equal")
        st.pyplot(fig)
    elif chart_type == "Box Plot":
        fig, ax = plt.subplots()
        ax.boxplot(data["Count"], vert=False, patch_artist=True)
        ax.set_yticklabels(['Topics'])
        st.pyplot(fig)
    st.markdown(
        "All visualizations are powered by your true topic modeling assignments from your processed dataset!"
    )

elif page == "Sentiment Analysis":
    st.header("Sentiment Analysis")
    SENTIMENT_LABELS = {0: "Negative", 1: "Positive",-1:"Negative"}   
    if "sent_texts" not in st.session_state:
        st.session_state.sent_texts = []
    if "sent_preds" not in st.session_state:
        st.session_state.sent_preds = []

    text_input = st.text_area("Enter text for sentiment classification (Any random text):")
    model_type = st.radio("Model:", ["LSTM", "RandomForest"])
    show_demo = st.checkbox("Show Example Charts/Graphs (Demo)", value=False)
    if st.button("Predict Sentiment"):
        if text_input.strip():
            if model_type == "RandomForest":
                model_file = 'model/Sentiment_analysis/random_forest_model.pkl'
                vectorizer_file = 'model/Sentiment_analysis/tfidf_vectorizer.pkl'
                if os.path.exists(model_file) and os.path.exists(vectorizer_file):
                    model = joblib.load(model_file)
                    vectorizer = joblib.load(vectorizer_file)
                    X = vectorizer.transform([text_input])
                    pred = model.predict(X)
                    pred_label = SENTIMENT_LABELS.get(pred[0], str(pred[0]))
                    st.success(f"Predicted Sentiment: {pred_label}")
                    st.session_state.sent_texts.append(text_input)
                    st.session_state.sent_preds.append(pred_label)
                else:
                    st.error("Random Forest model or vectorizer file not found.")
            else:
                model_file = 'model/Sentiment_analysis/lstm_model.h5'
                tokenizer_file = 'model/Sentiment_analysis/lstm_tokenizer.pkl'
                if os.path.exists(model_file) and os.path.exists(tokenizer_file):
                    try:
                        import tensorflow as tf
                        model = load_model(model_file, compile=False)
                        tokenizer = joblib.load(tokenizer_file)
                        seq = tokenizer.texts_to_sequences([text_input])
                        from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
                        seq = pad_sequences(seq, maxlen=10)  # use 10 for your model!
                        pred = model.predict(seq)
                        pred_class = int(np.argmax(pred))
                        pred_label = SENTIMENT_LABELS.get(pred_class, str(pred_class))
                        st.success(f"Predicted Sentiment class: {pred_label}")
                        st.session_state.sent_texts.append(text_input)
                        st.session_state.sent_preds.append(pred_label)
                    except Exception as e:
                        st.error("LSTM model could not be loaded. Please re-train and save the model using your current Keras/TensorFlow version. Details: " + str(e))
                else:
                    st.error("LSTM model or tokenizer file not found.")
        else:
            st.warning("Enter some text to classify.")

    if st.session_state.sent_texts:
        st.subheader("Recent Sentiment Predictions")
        for i, (text, pred) in enumerate(zip(st.session_state.sent_texts[-5:], st.session_state.sent_preds[-5:]), 1):
            st.markdown(f"**Sample {i}:** Predicted Sentiment: {pred}<br><small>{text[:150]}</small>", unsafe_allow_html=True)

    if st.session_state.sent_preds:
        st.subheader("Your Sentiment Prediction Charts")
        chart_type = st.radio("Select Chart Type", ["Pie Chart", "Bar Chart", "Confusion Matrix"], horizontal=True)
        preds = st.session_state.sent_preds
        unique_labels = sorted(list(set(preds)))
        labels, counts = np.unique(preds, return_counts=True)
        if chart_type == "Pie Chart":
            fig, ax = plt.subplots()
            ax.pie(counts, labels=unique_labels, autopct='%1.1f%%')
            st.pyplot(fig)
        elif chart_type == "Bar Chart":
            fig, ax = plt.subplots()
            ax.bar(unique_labels, counts)
            ax.set_ylabel("Counts")
            ax.set_title("Sentiment Prediction Distribution")
            st.pyplot(fig)
        elif chart_type == "Confusion Matrix":
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(preds, preds, labels=unique_labels)
            import seaborn as sns
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=unique_labels, yticklabels=unique_labels, ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True (self)')
            st.pyplot(fig)
    elif show_demo:
        st.subheader("Demo Sentiment Visualization Options")
        chart_type = st.radio("Select Chart Type", ["Pie Chart", "Bar Chart", "Confusion Matrix"], horizontal=True)
        y_true = ["Positive", "Negative", "Positive", "Negative"]
        y_pred = ["Positive", "Negative", "Positive", "Negative"]
        label_names = sorted(list(set(y_true + y_pred)))
        if chart_type == "Pie Chart":
            _, counts = np.unique(y_pred, return_counts=True)
            fig, ax = plt.subplots()
            ax.pie(counts, labels=label_names, autopct='%1.1f%%')
            st.pyplot(fig)
        elif chart_type == "Bar Chart":
            _, counts = np.unique(y_pred, return_counts=True)
            fig, ax = plt.subplots()
            ax.bar(label_names, counts)
            ax.set_ylabel("Counts")
            ax.set_title("Sentiment Prediction Distribution")
            st.pyplot(fig)
        elif chart_type == "Confusion Matrix":
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_true, y_pred, labels=label_names)
            import seaborn as sns
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=label_names, yticklabels=label_names, ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            st.pyplot(fig)

elif page == "Summarization":
    from transformers import pipeline
    st.markdown('<h1 style="margin-bottom:1rem">📝 Text Summarization</h1>', unsafe_allow_html=True)
    st.write("Upload or paste text to see how the system performs summarization. (Currently using a Transformer summarizer for demo.)")
    with st.form("summarize_form"):
        text_to_summarize = st.text_area("Enter text to summarize:", height=180)
        submitted = st.form_submit_button("Generate Summary")
    if submitted:
        if not text_to_summarize.strip():
            st.info("Enter text above and click 'Generate Summary'.")
        else:
            with st.spinner("Generating summary..."):
                summarizer = pipeline("summarization")
                summary = summarizer(text_to_summarize, max_length=80, min_length=30, do_sample=False)[0]['summary_text']
            st.markdown("---")
            st.success("**Summary:**")
            st.markdown(summary)

elif page == "Data Visualization":
    from wordcloud import WordCloud
    st.header("📊 Visualization Dashboard")
    df_test = pd.read_csv("data/processed/cnn_dailymail_test_summarized.csv")
    df_valid = pd.read_csv("data/processed/cnn_dailymail_valid_summarized.csv")
    if "text" not in df_test.columns:
        if "article" in df_test.columns:
            df_test = df_test.rename(columns={"article": "text"})
        elif "generated_summary" in df_test.columns:
            df_test = df_test.rename(columns={"generated_summary": "text"})
        elif "highlights" in df_test.columns:
            df_test = df_test.rename(columns={"highlights": "text"})
    if "text" not in df_valid.columns:
        if "article" in df_valid.columns:
            df_valid = df_valid.rename(columns={"article": "text"})
        elif "generated_summary" in df_valid.columns:
            df_valid = df_valid.rename(columns={"generated_summary": "text"})
        elif "highlights" in df_valid.columns:
            df_valid = df_valid.rename(columns={"highlights": "text"})
    df_test = analyze_data(df_test)
    df_valid = analyze_data(df_valid)

    tab1, tab2, tab3, tab4 = st.tabs(["Word Cloud", "Sentiment Distribution", "Topic Distribution", "Compare/Report"])
    with tab1:
        st.subheader("Test Set: Word Cloud")
        wc_test = WordCloud(width=800, height=400, background_color="white").generate(" ".join(df_test["text"].astype(str)))
        st.image(wc_test.to_array(), caption="Test Set Word Cloud", use_column_width=True)
        st.subheader("Valid Set: Word Cloud")
        wc_valid = WordCloud(width=800, height=400, background_color="white").generate(" ".join(df_valid["text"].astype(str)))
        st.image(wc_valid.to_array(), caption="Valid Set Word Cloud", use_column_width=True)
    with tab2:
        st.subheader("Test Set: Sentiment Distribution")
        fig, ax = plt.subplots()
        df_test["sentiment_label"].value_counts().plot(kind='bar', ax=ax, color=["green", "grey", "red"])
        ax.set_ylabel("Counts")
        st.pyplot(fig)

        st.subheader("Valid Set: Sentiment Distribution")
        fig, ax = plt.subplots()
        df_valid["sentiment_label"].value_counts().plot(kind='bar', ax=ax, color=["green", "grey", "red"])
        ax.set_ylabel("Counts")
        st.pyplot(fig)
    with tab3:
        st.subheader("Test Set: Topic Distribution")
        fig, ax = plt.subplots()
        df_test["assigned_topic"].value_counts().plot(kind='bar', ax=ax, color="blue")
        ax.set_ylabel("Counts")
        st.pyplot(fig)
        st.subheader("Valid Set: Topic Distribution")
        fig2, ax2 = plt.subplots()
        df_valid["assigned_topic"].value_counts().plot(kind='bar', ax=ax2, color="blue")
        ax2.set_ylabel("Counts")
        st.pyplot(fig2)
    with tab4:
        st.subheader("Compare Topic Distributions")
        label_set = sorted(set(df_test["assigned_topic"]) | set(df_valid["assigned_topic"]))
        t1 = df_test["assigned_topic"].value_counts().reindex(label_set, fill_value=0)
        t2 = df_valid["assigned_topic"].value_counts().reindex(label_set, fill_value=0)
        fig, ax = plt.subplots()
        ax.bar(np.arange(len(label_set))-0.2, t1, width=0.4, label="Test")
        ax.bar(np.arange(len(label_set))+0.2, t2, width=0.4, label="Valid")
        ax.set_xticks(np.arange(len(label_set)))
        ax.set_xticklabels(label_set)
        ax.set_ylabel("Counts")
        ax.legend()
        st.pyplot(fig)

        st.subheader("Compare Sentiment Distributions")
        sent_labels = sorted(set(df_test["sentiment_label"]) | set(df_valid["sentiment_label"]))
        v1 = df_test["sentiment_label"].value_counts().reindex(sent_labels, fill_value=0)
        v2 = df_valid["sentiment_label"].value_counts().reindex(sent_labels, fill_value=0)
        fig2, ax2 = plt.subplots()
        ax2.bar(np.arange(len(sent_labels))-0.2, v1, width=0.4, label="Test")
        ax2.bar(np.arange(len(sent_labels))+0.2, v2, width=0.4, label="Valid")
        ax2.set_xticks(np.arange(len(sent_labels)))
        ax2.set_xticklabels(sent_labels)
        ax2.set_ylabel("Counts")
        ax2.legend()
        st.pyplot(fig2)

        st.markdown("""
        **Automated Report & Insights:**  
        - Generated word clouds show key themes for both test and validation sets.
        - Sentiment and topic distributions above let you compare news set coverage and emotional tone.
        - For deeper insights, export these charts or access classwise frequency tables.
        """)

elif page == "Live Demo":
    from transformers import pipeline
    st.header("Live Demo")
    st.markdown("""
    <h3>Input Section</h3>
    Choose input method, sentiment model, and analyze for topic/sentiment.
    """, unsafe_allow_html=True)

    input_method = st.radio(
        "Choose input method:",
        ("Enter Text", "Upload Document", "Reddit Post Link"),
        horizontal=True
    )
    text_input = ""
    reddit_link = ""
    df_uploaded = None
    if input_method == "Enter Text":
        text_input = st.text_area("Enter text below:")
    elif input_method == "Upload Document":
        uploaded_file = st.file_uploader("Upload a CSV or TXT file", type=["csv", "txt"])
        if uploaded_file is not None:
            df_uploaded = pd.read_csv(uploaded_file)
            if "text" not in df_uploaded.columns:
                if "article" in df_uploaded.columns:
                    df_uploaded = df_uploaded.rename(columns={"article": "text"})
                elif "generated_summary" in df_uploaded.columns:
                    df_uploaded = df_uploaded.rename(columns={"generated_summary": "text"})
                elif "highlights" in df_uploaded.columns:
                    df_uploaded = df_uploaded.rename(columns={"highlights": "text"})
        else:
            text_input = uploaded_file.read().decode("utf-8")
    else:
        reddit_link = st.text_input("Enter Reddit post link:")

    model_type = st.radio("Choose Sentiment Model", ["Random Forest (TF-IDF)", "LSTM (Deep Learning)"], horizontal=True)
    generate_summary = st.checkbox("Also generate summary using Gemini 2.0 Flash (stub)")
    if st.button("Analyze"):
        if input_method == "Enter Text" and text_input.strip():
            to_analyze = [text_input]
        elif input_method == "Upload Document" and df_uploaded is not None:
            if "text" in df_uploaded.columns:
                to_analyze = df_uploaded["text"].astype(str).tolist()
            else:
                st.error("For CSV, must have a 'text', 'article', 'generated_summary', or 'highlights' column.")
                to_analyze = []
        elif input_method == "Reddit Post Link" and reddit_link:
            st.info("Reddit ingestion is a placeholder in this demo.")
            to_analyze = ["Reddit post content for: " + reddit_link]
        else:
            st.error("Please enter valid input for your selection above!")
            to_analyze = []

        if len(to_analyze) > 0:
            st.subheader("Sentiment Results")
            preds = []
            if "Random Forest" in model_type:
                model_file = 'model/Sentiment_analysis/random_forest_model.pkl'
                vectorizer_file = 'model/Sentiment_analysis/tfidf_vectorizer.pkl'
                if os.path.exists(model_file) and os.path.exists(vectorizer_file):
                    model = joblib.load(model_file)
                    vectorizer = joblib.load(vectorizer_file)
                    X = vectorizer.transform(to_analyze)
                    preds = model.predict(X)
                else:
                    st.error("Random Forest model/vectorizer file not found.")
            elif "LSTM" in model_type:
                model_file = 'model/Sentiment_analysis/lstm_model.h5'
                tokenizer_file = 'model/Sentiment_analysis/lstm_tokenizer.pkl'
                if os.path.exists(model_file) and os.path.exists(tokenizer_file):
                    model = load_model(model_file)
                    tokenizer = joblib.load(tokenizer_file)
                    seqs = tokenizer.texts_to_sequences(to_analyze)
                    seqs = pad_sequences(seqs, maxlen=100)
                    preds = [np.argmax(x) for x in model.predict(seqs)]
                else:
                    st.error("LSTM model or tokenizer file not found.")

            if len(preds) > 0:
                for i, (text, pred) in enumerate(zip(to_analyze, preds)):
                    st.markdown(f"**Sample {i+1}:** Predicted Sentiment: {pred}<br><small>{text[:150]}</small>", unsafe_allow_html=True)
                if len(preds) > 1:
                    st.subheader("Chart Visualization")
                    chart_options = st.multiselect("Choose charts to show:", ["Pie Chart", "Bar Chart"])
                    labels, counts = np.unique(preds, return_counts=True)
                    if "Pie Chart" in chart_options:
                        fig, ax = plt.subplots()
                        ax.pie(counts, labels=labels, autopct='%1.1f%%')
                        st.pyplot(fig)
                    if "Bar Chart" in chart_options:
                        fig, ax = plt.subplots()
                        ax.bar(labels, counts)
                        ax.set_ylabel("Counts")
                        ax.set_title("Sentiment Prediction Distribution")
                        st.pyplot(fig)

            if generate_summary and len(to_analyze) == 1:
                st.subheader("Gemini 2.0 Flash Summary (Demo)")
                try:
                    summarizer = pipeline("summarization")
                    summary_results = summarizer(to_analyze[0], max_length=80, min_length=30, do_sample=False)
                    if summary_results and isinstance(summary_results, list) and 'summary_text' in summary_results[0]:
                        st.markdown("**Summary:** " + summary_results[0]['summary_text'])
                    else:
                        st.warning("No summary returned.")
                except Exception as e:
                    st.error(f"Summary generation failed: {e}")

elif page == "About":
    st.header("About AI Narrative Nexus")
    st.markdown("""
AI Narrative Nexus converts unstructured text into insights using:

- **Topic Modeling** (LDA / NMF)
- **Sentiment Analysis**
- **Text Summarization**
- **Interactive Visualizations**

---
**Tech Stack**:
Python   🤗 Transformers   🍰 NLTK   🎈 Streamlit   🟨 Matplotlib   ☁️ WordCloud

---
**Modules/Replace Placeholders:**
- `/src/preprocess`
- `/src/topics`
- `/src/sentiment`
- `/src/summarize`
- `/src/viz`

---
Team: Your Name | Project: Infosys Internship 2025  
""", unsafe_allow_html=True)

    with open("AI_Narrative_Nexus.pdf", "rb") as f:
        pdf_data = f.read()
    st.download_button(
        label="⬇️ Download Project Report: AI_Narrative_Nexus.pdf",
        data=pdf_data,
        file_name="AI-Narrative-Nexus.pdf",
        mime="application/pdf"
    )
