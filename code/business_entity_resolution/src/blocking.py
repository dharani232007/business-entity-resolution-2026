import re
import pandas as pd
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def extract_primary_token(name: str) -> str:
    tokens = re.findall(r"\b[a-z0-9]{3,}\b", str(name).lower())
    return tokens[0] if tokens else ""

def generate_candidates_fast(s1_df: pd.DataFrame, s23_df: pd.DataFrame, top_k: int = 15) -> dict:
    candidates = {eid: [] for eid in s1_df["entity_id"]}
    countries = s1_df["country"].dropna().unique()
    
    for country in countries:
        s1_sub = s1_df[s1_df["country"] == country].reset_index(drop=True)
        s23_sub = s23_df[s23_df["country"] == country].reset_index(drop=True)
        
        if s1_sub.empty or s23_sub.empty:
            continue
            
        token_index = defaultdict(list)
        for idx, name in enumerate(s23_sub["clean_name"]):
            token = extract_primary_token(name)
            if token:
                token_index[token].append(idx)
                
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4))
        
        for i, row in s1_sub.iterrows():
            s1_id = row["entity_id"]
            s1_name = row["clean_name"]
            token = extract_primary_token(s1_name)
            
            matched_indices = token_index.get(token, [])
            if not matched_indices:
                continue
                
            if len(matched_indices) > 200:
                matched_indices = matched_indices[:200]
                
            candidate_names = s23_sub.loc[matched_indices, "clean_name"].tolist()
            
            try:
                tfidf_mat = vectorizer.fit_transform([s1_name] + candidate_names)
                sims = cosine_similarity(tfidf_mat[0:1], tfidf_mat[1:]).flatten()
                ranked_pairs = sorted(zip(matched_indices, sims), key=lambda x: x[1], reverse=True)[:top_k]
                candidates[s1_id] = [s23_sub.loc[idx, "entity_id"] for idx, score in ranked_pairs if score >= 0.25]
            except Exception:
                continue
                
    return candidates
