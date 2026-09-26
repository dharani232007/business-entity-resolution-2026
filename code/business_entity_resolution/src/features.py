from rapidfuzz import fuzz

def compute_pair_features(s1_name: str, s1_addr: str, cand_name: str, cand_addr: str) -> dict:
    """Computes string similarity metrics between an S1 entity and a candidate."""
    return {
        "name_ratio": fuzz.ratio(s1_name, cand_name),
        "name_token_sort": fuzz.token_sort_ratio(s1_name, cand_name),
        "name_token_set": fuzz.token_set_ratio(s1_name, cand_name),
        "addr_partial": fuzz.partial_ratio(s1_addr, cand_addr),
        "addr_token_set": fuzz.token_set_ratio(s1_addr, cand_addr)
    }

def is_confident_match(feats: dict) -> bool:
    """
    Applies high-precision matching rules to optimize for the macro F0.5 metric.
    Penalizes false merges twice as heavily as missed links.
    """
    # Exact or near-identical name match
    if feats["name_token_sort"] >= 95:
        return True
    # Strong name match with consistent address signals
    if feats["name_token_sort"] >= 80 and feats["addr_partial"] >= 65:
        return True
    if feats["name_ratio"] >= 85 and feats["addr_token_set"] >= 60:
        return True
    return False
