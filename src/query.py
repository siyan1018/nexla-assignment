from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re




def clean_text(text):
    text = text.replace('\n', ' ')
    text = re.sub(r'[^a-zA-Z0-9.,\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def extract_keywords(question):
    stopwords = {
        "what", "is", "the", "in", "on", "at", "a", "an", "of",
        "to", "and", "or", "how", "why", "when", "where", "which",
        "do", "does", "did", "was", "were", "are", "be", "been",
        "can", "could", "would", "should", "has", "have", "had",
    }
    words = re.sub(r'[^a-z0-9\s]', '', question.lower()).split()
    return [w for w in words if w not in stopwords and len(w) > 1]



def is_near_duplicate(text_a, text_b, threshold=0.85):
    """
    Jaccard similarity on word sets.
    Fast, zero-dependency check that catches copy-paste chunks
    and lightly rephrased duplicates.
    """
    set_a = set(text_a.lower().split())
    set_b = set(text_b.lower().split())
    if not set_a or not set_b:
        return False
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return (intersection / union) >= threshold


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def hybrid_score(sim_score, text, keywords, keyword_weight=0.35):
    """
    Blend cosine similarity with a keyword-coverage bonus.

    keyword_weight controls how much keyword hits can lift the score.
    A chunk that matches all keywords gets the full bonus; partial
    matches are proportional.
    """
    if not keywords:
        return float(sim_score)

    text_lower = text.lower()
    hits = sum(1 for kw in keywords if kw in text_lower)
    keyword_bonus = hits / len(keywords)          
    sim_weight = 1.0 - keyword_weight

    return sim_weight * float(sim_score) + keyword_weight * keyword_bonus


  

def mmr_rerank(candidates, embeddings, lambda_param=0.6, top_k=3):
    """
    Picks results that balance relevance (high hybrid score) and
    diversity (low similarity to already-selected chunks).

    lambda_param = 1.0  → pure relevance (no diversity)
    lambda_param = 0.0  → pure diversity (ignore relevance)
    0.5–0.7 is a good practical range.

    candidates : list of dicts with keys 'idx', 'score'
    embeddings : the full embedding matrix passed into query_documents
    """
    if not candidates:
        return []

    selected_indices = []     # positions in `candidates` list
    selected_emb_indices = [] # positions in `embeddings` matrix

    for _ in range(min(top_k, len(candidates))):
        best_pos = None
        best_mmr = -np.inf

        for pos, cand in enumerate(candidates):
            if pos in selected_indices:
                continue

            relevance = cand["score"]

            # Similarity to already-selected chunks
            if selected_emb_indices:
                sel_embs = embeddings[selected_emb_indices]
                cand_emb = embeddings[cand["idx"]]
                max_sim = cosine_similarity(
                    cand_emb.reshape(1, -1), sel_embs
                ).max()
            else:
                max_sim = 0.0

            mmr = lambda_param * relevance - (1 - lambda_param) * max_sim

            if mmr > best_mmr:
                best_mmr = mmr
                best_pos = pos

        if best_pos is None:
            break

        selected_indices.append(best_pos)
        selected_emb_indices.append(candidates[best_pos]["idx"])

    return [candidates[i] for i in selected_indices]




def query_documents(
    question,
    docs,
    embeddings,
    vectorizer,
    top_k=3,
    min_chunk_len=80,
    max_per_source=2,
    mmr_lambda=0.6,
    keyword_weight=0.35,
    near_dup_threshold=0.85,
):
    """
    Retrieve the top_k most relevant *and* diverse chunks for a question.

    Improvements over the original
    ───────────────────────────────
    1. Hybrid scoring  – cosine similarity + keyword-coverage bonus.
    2. Source capping  – at most `max_per_source` chunks per document/page,
                         preventing any single source from dominating.
    3. MMR re-ranking  – Maximal Marginal Relevance picks results that are
                         relevant but not redundant with each other.
    4. Deduplication   – near-duplicate chunks (Jaccard ≥ threshold) are
                         skipped before MMR runs.

    Parameters
    ----------
    question          : raw user query string
    docs              : list of dicts with keys 'text', 'source', 'page'
    embeddings        : sparse/dense matrix (n_docs × n_features)
    vectorizer        : fitted sklearn vectorizer
    top_k             : number of results to return
    min_chunk_len     : discard chunks shorter than this (characters)
    max_per_source    : max chunks allowed from same (source, page) pair
    mmr_lambda        : MMR trade-off; higher = more relevance-focused
    keyword_weight    : share of score coming from keyword coverage
    near_dup_threshold: Jaccard threshold above which a chunk is a duplicate
    """

    # 1. Embed query and compute cosine similarity against all chunks
    question_vec = vectorizer.transform([question])
    similarities = cosine_similarity(question_vec, embeddings).flatten()

    keywords = extract_keywords(question)

    # 2. Build candidate pool: filter garbage, apply hybrid score
    candidates = []
    source_counts = {}   # (source, page) → count

    for idx, sim_score in enumerate(similarities):
        text = docs[idx]["text"]

        # Drop chunks that are too short to be useful
        if len(text.strip()) < min_chunk_len:
            continue

        score = hybrid_score(sim_score, text, keywords, keyword_weight)
        candidates.append({"idx": idx, "score": score})

    if not candidates:
        return []

    # 3. Sort by hybrid score (best first)
    candidates.sort(key=lambda x: x["score"], reverse=True)

    
    pre_selection = []
    seen_texts = []

    for cand in candidates[: top_k * 8]:
        doc = docs[cand["idx"]]
        key = (doc["source"], doc["page"])
        text = clean_text(doc["text"])

        # Enforce per-source cap
        if source_counts.get(key, 0) >= max_per_source:
            continue

        if any(is_near_duplicate(text, seen, near_dup_threshold) for seen in seen_texts):
            continue

        source_counts[key] = source_counts.get(key, 0) + 1
        seen_texts.append(text)
        pre_selection.append(cand)

    # 5. MMR re-rank for diversity
    final_candidates = mmr_rerank(
        pre_selection, embeddings, lambda_param=mmr_lambda, top_k=top_k
    )

    # 6. Build output
    results = []
    for cand in final_candidates:
        doc = docs[cand["idx"]]
        results.append({
            "text": clean_text(doc["text"]),
            "source": doc["source"],
            "page": doc["page"],
            "score": round(cand["score"], 4),   # handy for debugging
        })

    return results
