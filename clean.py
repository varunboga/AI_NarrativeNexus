import os
import pandas as pd
import re

def clean_body_classification(raw_text: str) -> str:
    if not raw_text or pd.isna(raw_text):
        return ""
    parts = re.split(r"\n\s*\n", raw_text, maxsplit=1)
    body = parts[1] if len(parts) > 1 else parts[0]
    cleaned_lines = []
    for line in body.splitlines():
        if re.match(r"^\s*(from|subject|path|organization|lines|newsgroups|message-id|keywords|last-modified|version|distribution|summary|sender|references|nntp-posting-host|article-id|followup-to|content-type|content-transfer-encoding)\s*:", line, re.I):
            continue
        if re.match(r"^(\-\-+|__+|\*\*+)\s*$", line.strip()):
            break
        if line.strip().startswith((">", "|")):
            continue
        if re.search(r"(writes:|wrote:|In article\s*<.*?>)", line, re.I):
            continue
        cleaned_lines.append(line)
    body = "\n".join(cleaned_lines)
    body = re.sub(r"\b\S+@\S+\b", " ", body)
    body = re.sub(r"http\S+|www\.\S+", " ", body)
    body = re.sub(r"<[^>]+>", " ", body)
    body = re.sub(r"\d+", " ", body)
    body = re.sub(r"[^a-zA-Z\s]", " ", body)
    body = re.sub(r"\s{2,}", " ", body)
    body = re.sub(r"\n{3,}", "\n\n", body)
    return body.lower().strip()

def sanitize_for_excel(value: str) -> str:
    if isinstance(value, str) and value and value[0] in ('=', '+', '-', '@'):
        return "'" + value
    return value

def convert_20ng_dataset(root_folder, output_excel, output_csv, max_files=None):
    os.makedirs(root_folder, exist_ok=True)
    data = []
    for category in sorted(os.listdir(root_folder)):
        category_path = os.path.join(root_folder, category)
        if os.path.isdir(category_path):
            print(f" Processing category: {category}")
            for i, filename in enumerate(os.listdir(category_path)):
                if max_files and i >= max_files:
                    break
                file_path = os.path.join(category_path, filename)
                try:
                    with open(file_path, "r", encoding="latin1") as f:
                        raw_text = f.read()
                        body = clean_body_classification(raw_text)
                        if body:
                            data.append({
                                "filename": filename,
                                "category": category,
                                "text": body
                            })
                except Exception as e:
                    print(f" Skipping {file_path}: {e}")
    df = pd.DataFrame(data)
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].map(sanitize_for_excel)
    df.to_excel(output_excel, index=False, engine="openpyxl")
    df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f" Saved {len(df)} rows across {df['category'].nunique()} categories")
    print(f"   Excel: {output_excel}")
    print(f"   CSV  : {output_csv}")

convert_20ng_dataset(
    root_folder = "C:/Users/Varun Boga/OneDrive/Desktop/AI Narrative Nexus/data/req_data/20news-18828/",
    output_excel="C:/Users/Varun Boga/OneDrive/Desktop/AI Narrative Nexus/data/req_data/20news-18828/20news_18828_clean.xlsx",
    output_csv="C:/Users/Varun Boga/OneDrive/Desktop/AI Narrative Nexus/data/req_data/20news-18828/20news_18828_clean.csv",
    max_files=100
)
