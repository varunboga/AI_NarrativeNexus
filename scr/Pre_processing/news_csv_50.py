import os
import pandas as pd
import re

def extract_body(text):
    parts = re.split(r"\n\s*\n", text, maxsplit=1)
    body = parts[1] if len(parts) > 1 else parts[0]
    cleaned_lines = []
    for line in body.splitlines():
        if re.match(r"^(Archive-name|From|Subject|Path|Xref|Organization|Lines|Newsgroups|Message-ID|Keywords):", line, re.I):
            continue
        if line.strip().startswith(">"):
            continue
        cleaned_lines.append(line)
    body_text = "\n".join(cleaned_lines).strip()
    body_text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", body_text)
    return body_text

def sanitize_for_excel(value: str) -> str:
    if isinstance(value, str) and value and value[0] in ('=', '+', '-', '@'):
        return "'" + value
    return value

def convert_20ng_to_excel(root_folder, output_excel, max_files=50):
    data = []
    for category in sorted(os.listdir(root_folder)):
        category_path = os.path.join(root_folder, category)
        if os.path.isdir(category_path):
            print(f" Processing category: {category}")
            for i, filename in enumerate(os.listdir(category_path)):
                if i >= max_files:
                    break
                file_path = os.path.join(category_path, filename)
                try:
                    with open(file_path, 'r', encoding='latin1') as f:
                        raw_text = f.read()
                        body = extract_body(raw_text)
                        if body:
                            data.append({
                                "filename": filename,
                                "category": category,
                                "text": body
                            })
                except Exception as e:
                    print(f" Skipping {file_path}: {e}")
    df = pd.DataFrame(data)
    df = df.applymap(sanitize_for_excel)
    df.to_excel(output_excel, index=False, engine="openpyxl")
    print(f" Saved {len(df)} rows across {df['category'].nunique()} categories to {output_excel}")

convert_20ng_to_excel(
    root_folder="C:/Users/Varun Boga/OneDrive/Desktop/AI Narrative Nexus/data/req_data/20news-18828",
    output_excel="C:/Users/Varun Boga/OneDrive/Desktop/AI Narrative Nexus/data/req_data/20news-18828/20news_18828_clean_50.xlsx",
    max_files=50
)
