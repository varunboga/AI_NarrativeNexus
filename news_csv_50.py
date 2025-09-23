import os
import pandas as pd
import re

def extract_body(text):
    """
    Extract the body of a 20NG post:
    - Remove headers
    - Drop lines like Archive-name, Subject, From, etc.
    - Skip quoted lines (>)
    """
    # Split headers vs body
    parts = re.split(r"\n\s*\n", text, maxsplit=1)
    body = parts[1] if len(parts) > 1 else parts[0]

    # Drop metadata lines
    cleaned_lines = []
    for line in body.splitlines():
        if re.match(r"^(Archive-name|From|Subject|Path|Xref|Organization|Lines|Newsgroups|Message-ID|Keywords):", line, re.I):
            continue
        if line.strip().startswith(">"):  # quoted text
            continue
        cleaned_lines.append(line)

    # Remove illegal Excel characters
    body_text = "\n".join(cleaned_lines).strip()
    body_text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", body_text)

    return body_text


def sanitize_for_excel(value: str) -> str:
    """Prevent Excel from misinterpreting text as a formula."""
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


# Run
convert_20ng_to_excel("req_data\\20news_18828", "20news_18828_clean_50.xlsx", max_files=50)