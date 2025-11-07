import pandas as pd
import re

def clean_body(raw_text):
    if pd.isna(raw_text):
        return ""
    parts = re.split(r"\n\s*\n", raw_text, maxsplit=1)
    body = parts[1] if len(parts) > 1 else parts[0]
    cleaned_lines = []
    for line in body.splitlines():
        if re.match(
            r"^(archive-name|from|subject|path|xref|organization|lines|newsgroups|message-id|keywords|last-modified|version):",
            line,
            re.I,
        ):
            continue
        if line.strip().startswith((">", "|")):
            continue
        if line.strip().startswith("--"):
            break
        if re.search(r"In article\s*<.*?>", line, re.I):
            continue
        if re.search(r"writes:|wrote:", line, re.I):
            continue
        cleaned_lines.append(line)
    body = "\n".join(cleaned_lines)
    body = re.sub(r"\S+@\S+", " ", body)
    body = re.sub(r"http\S+|www\.\S+", " ", body)
    body = re.sub(r"<[^>]+>", " ", body)
    body = re.sub(r"[^a-zA-Z0-9\s\.\,\!\?]", " ", body)
    body = re.sub(r"\n{2,}", "\n", body)
    body = re.sub(r"\s{2,}", " ", body)
    body = body.lower().strip()
    body = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", body)
    return body

input_file = "C:/Users/Varun Boga/OneDrive/Desktop/AI Narrative Nexus/data/req_data/20news-18828/20news_18828_clean_50.xlsx"
output_file = "C:/Users/Varun Boga/OneDrive/Desktop/AI Narrative Nexus/data/req_data/20news-18828/20news_18828_final_50.xlsx"

df = pd.read_excel(input_file, engine="openpyxl")
df["text"] = df["text"].apply(clean_body)
df.to_excel(output_file, index=False, engine="openpyxl")

print(f" Final dataset saved: {len(df)} rows, cleaned text only")
