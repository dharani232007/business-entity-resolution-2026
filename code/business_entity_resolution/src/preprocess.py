import re
import pandas as pd

def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower().replace("&", " and ")
    legal_map = {
        r"\bcorp(\.|\b)": "corporation",
        r"\binc(\.|\b)": "incorporated",
        r"\bltd(\.|\b)": "limited",
        r"\bpvt(\.|\b)": "private",
        r"\bco(\.|\b)": "company",
        r"\bllc(\.|\b)": "limited liability company",
        r"\brd(\.|\b)": "road",
        r"\bst(\.|\b)": "street",
    }
    for pattern, repl in legal_map.items():
        text = re.sub(pattern, repl, text)
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())

def load_source(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, sep="\t", dtype=str).fillna("")
    df["clean_name"] = df["business_name"].apply(normalize_text)
    df["clean_address"] = df["business_address"].apply(normalize_text)
    df["country"] = df["country"].str.strip().str.upper()
    return df
