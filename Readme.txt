# AI Narrative Nexus

## Description

**AI Narrative Nexus** transforms raw textual data into actionable insights using state-of-the-art Natural Language Processing (NLP):

- **Topic Modeling (LDA / NMF)**
- **Sentiment Analysis**
- **Text Summarization**
- **Interactive Data Visualizations**

Upload your data, analyze it, and visualize patterns instantly. Perfect for reports, social media, feedback, and more.

---

## Getting Started

### Dependencies

- Python 3.8+
- Streamlit
- Transformers
- sklearn
- NLTK
- matplotlib
- wordcloud
- joblib
- pandas

> *See `requirements.txt` for full list*

### Installing

Clone the repository and install the requirements:

```bash
git clone https://github.com/your_username/ai-narrative-nexus.git
cd ai-narrative-nexus
pip install -r requirements.txt
```

Place your trained models and sample data into their corresponding folders as per the project structure.

### Running the Program

To start the Streamlit dashboard:

```bash
streamlit run app.py
```

#### Typical Steps

- Select a module in the sidebar (Home, Upload & Analyze, etc.)
- Upload your data or paste your text
- View summaries, topics, and visualizations live!

---

## Folder Structure

```plaintext
ai-narrative-nexus/
├── app.py
├── /src
│   ├── preprocess
│   ├── topics
│   ├── sentiment
│   ├── summarize
│   └── viz
├── /model
├── /data
└── requirements.txt
```

---

## Help

- For errors with file uploads, ensure your CSV has a `text` column.
- If model files are missing, reload or re-download from `/model`.
- For troubleshooting, consult the `AI_Narrative_Nexus.pdf` report.

---

## Authors

- Your Name (your.email@example.com)
- Project: Infosys Internship 2025

---

## Version History

- **0.1** – Initial Release: Major modules and interactive dashboard complete
- **0.2** – [In Progress] Sentiment model upgrade, topic labeling, automated reporting

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

*AI Narrative Nexus: automate your text analytics—all in one dashboard!*
