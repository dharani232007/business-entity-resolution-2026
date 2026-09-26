import os
import sys
import pandas as pd

from preprocess import normalize_text, load_source
from blocking import generate_candidates_fast
from features import compute_pair_features, is_confident_match

def run_inference(test_dir="dataset/test", output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading test datasets...")
    s1 = load_source(f"{test_dir}/test_source1.tsv")
    s2 = load_source(f"{test_dir}/test_source2.tsv")
    s3 = load_source(f"{test_dir}/test_source3.tsv")
    
    # Combine S2 and S3 for lookup pool
    s23 = pd.concat([s2, s3], ignore_index=True)
    print(f"Total S1 entities: {len(s1):,} | Candidate pool (S2+S3): {len(s23):,}")
    
    print("Generating candidate pairs via country & token blocking...")
    candidates = generate_candidates_fast(s1, s23, top_k=20)
    
    # 1. Write candidate_pairs.tsv
    cand_rows = []
    for eid in s1["entity_id"]:
        cand_list = candidates.get(eid, [])
        # Deduplicate while preserving order
        seen = set()
        clean_cands = [c for c in cand_list if not (c in seen or seen.add(c))]
        cand_rows.append({
            "source1_entity_id": eid,
            "candidate_entity_ids": ",".join(clean_cands)
        })
    cand_df = pd.DataFrame(cand_rows)
    cand_path = f"{output_dir}/candidate_pairs.tsv"
    cand_df.to_csv(cand_path, sep="\t", index=False)
    print(f"Saved: {cand_path} ({len(cand_df):,} rows)")
    
    # Pre-index lookups for fast retrieval
    s1_lookup = s1.set_index("entity_id").to_dict("index")
    s23_lookup = s23.set_index("entity_id").to_dict("index")
    
    print("Classifying matches with high-precision gating...")
    match_rows = []
    for row in cand_rows:
        s1_id = row["source1_entity_id"]
        cand_str = row["candidate_entity_ids"]
        cand_ids = cand_str.split(",") if cand_str else []
        
        matches = []
        s1_meta = s1_lookup.get(s1_id, {})
        s1_name = s1_meta.get("clean_name", "")
        s1_addr = s1_meta.get("clean_address", "")
        
        for cid in cand_ids:
            cand_meta = s23_lookup.get(cid, {})
            if not cand_meta:
                continue
            feats = compute_pair_features(
                s1_name, s1_addr,
                cand_meta.get("clean_name", ""),
                cand_meta.get("clean_address", "")
            )
            if is_confident_match(feats):
                matches.append(cid)
                
        # Deduplicate confirmed matches
        seen_m = set()
        clean_matches = [m for m in matches if not (m in seen_m or seen_m.add(m))]
        match_rows.append({
            "source1_entity_id": s1_id,
            "matched_entity_ids": ",".join(clean_matches)
        })
        
    match_df = pd.DataFrame(match_rows)
    match_path = f"{output_dir}/matching_results.tsv"
    match_df.to_csv(match_path, sep="\t", index=False)
    print(f"Saved: {match_path} ({len(match_df):,} rows)")

if __name__ == "__main__":
    run_inference()
