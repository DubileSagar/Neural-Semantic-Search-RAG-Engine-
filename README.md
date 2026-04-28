# Neural Semantic Search / RAG Engine 

> Semantic search over the 20 Newsgroups dataset with fuzzy clustering, a first-principles cache layer, and a live FastAPI service — built for the Trademarkia AI/ML Engineer assignment.

---

## Tech Stack

| Component | Choice | Why |
|-----------|--------|-----|
| Embeddings | `all-MiniLM-L6-v2` | 384-dim vectors, best speed/accuracy ratio for semantic similarity. Smaller than BERT-base (768d) or Ada-002 (1536d) — halves memory and speeds up every cache dot-product |
| Vector DB | ChromaDB (local) | No API keys, no external servers. SQLite-backed persistent storage for ~18k docs with cosine search built-in |
| Clustering | Fuzzy C-Means (`skfuzzy`) | Soft membership — each doc gets a probability distribution across clusters, not a hard label. Reflects real-world topic overlap |
| Cache | Custom (no Redis) | Dictionary bucketed by nearest cluster. Lookup is O(n/k) instead of O(n) — clusters power the index, not just labels |
| API | FastAPI + uvicorn | Single-command startup, Pydantic schemas, auto Swagger docs at `/docs` |
| Frontend | Vanilla HTML/CSS/JS | Glassmorphism UI with live cache dashboard, UI/JSON toggle, threshold slider |

---

## Architecture

```
POST /query
    │
    ├─ 1. Embed query (all-MiniLM-L6-v2, 384d)
    │
    ├─ 2. Find nearest cluster (argmin centroid distance in PCA-50 space)
    │
    ├─ 3. Lookup cache bucket[cluster_id]
    │         │
    │    Hit ─┤─ Miss
    │         │       │
    │    return       ├─ search ChromaDB (top-5 cosine matches)
    │    cached       ├─ store result in bucket
    │    result       └─ return result
    │
    └─ Response: { query, cache_hit, matched_query, similarity_score, result, dominant_cluster }
```

---

## Design Decisions


**Why `all-MiniLM-L6-v2`?**
Purpose-built for semantic similarity. 384 dims vs 768 (BERT) or 1536 (Ada) — smaller vectors mean faster dot-product comparisons inside the cache on every request.

**Why ChromaDB over Pinecone/Milvus?**
All external vector DBs require cloud accounts or running a separate database server. For 18k documents, that's infrastructure overkill. ChromaDB gives persistent local cosine search with zero setup.

**Why PCA before clustering?**
Clustering in 384 dimensions hits the curse of dimensionality — all pairwise distances converge and cluster boundaries become meaningless. PCA to 50 dims retains ~90% variance while making FCM numerically stable.

**Why is the cluster count not hardcoded?**
`setup.py` sweeps k=5 to k=25, computes Fuzzy Partition Coefficient (FPC) for each, and picks the peak. For this dataset, optimal k=5.

**Why nearest-centroid for query assignment?**
The FCM membership formula degenerates to uniform probabilities when query distances to all centroids are similar in scale. `argmin(distances)` always gives a deterministic, meaningful cluster assignment.

**Why is the cache bucketed by cluster?**
Flat cache → O(n) lookups. Bucketed by cluster → O(n/k). At 10k cached queries with k=5 clusters, that's a 5x speedup. The clustering actively powers cache performance.

---

## Project Structure

```
trademarkia/
├── main.py                      # entry point — uvicorn main:app
├── setup.py                     # one-time: embed, cluster, index corpus
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py         # load + clean 20 Newsgroups dataset
│   ├── embeddings.py            # sentence-transformers + ChromaDB
│   ├── clustering.py            # Fuzzy C-Means + PCA + boundary analysis
│   ├── cache.py                 # cluster-bucketed semantic cache
│   └── api.py                   # FastAPI endpoints + lifespan startup
│
├── static/
│   ├── index.html               # frontend UI
│   ├── app.js                   # search logic + UI/JSON toggle
│   └── styles.css               # glassmorphism theme
│
├── notebooks/
│   └── analysis.ipynb           # exploratory analysis notebook
│
├── data/                        # placeholder (dataset auto-downloaded)
│
└── vector_store/                # generated after running setup.py
    ├── chroma.sqlite3           # ChromaDB persistent store
    └── clustering.pkl           # PCA model + FCM centroids + memberships
```
## 🔍 Demo Query Walkthrough

---

### 📦 Cluster 0 — GPU / Hardware Topic

All these queries land in **Cluster 0**. Only 1 cache entry is created — every rephrase hits it.

| # | Query | Cache | Note |
|---|-------|-------|------|
| 1 | `best graphics card for gaming` | 🔴 MISS | Cache empty, entry created in Cluster 0 |
| 2 | `best graphics card for gaming` | 🟢 HIT | Exact repeat |
| 3 | `top GPU recommendations for PC games` | 🟢 HIT | Different words, same meaning |
| 4 | `which GPU should I buy under $500` | 🟢 HIT | More specific, still hits |
| 5 | `best video card for high fps gaming` | 🟢 HIT | All 5 words different — still hits |

> 1 cache entry served 4 hits. Every GPU-buying rephrase collapsed onto the same stored result.

---

### 🌌 Cluster 9 — Space / Science Topic (completely different)

| # | Query | Cache | Note |
|---|-------|-------|------|
| 6 | `NASA Mars mission and space exploration` | 🔴 MISS | New topic → new cluster, new entry created |
| 7 | `latest updates on Mars rover` | 🟢 HIT | Hits Cluster 9 — Cluster 0 never touched |

> The cache now has 2 entries across 2 clusters. Each query only searches its own cluster bucket.

---

### 📊 Final Stats
```json
{
  "total_entries": 2,
  "hit_count": 5,
  "miss_count": 2,
  "hit_rate": 0.714
}
```
---

## Quick Start

### Local (Recommended)

```bash
# 1. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate        # windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. One-time setup: embed corpus + cluster (~5-10 min)
python setup.py

# 4. Start the server
uvicorn main:app --port 8000
```

Open [http://localhost:8000](http://localhost:8000)

### Docker

```bash
docker-compose up --build
# server at http://localhost:8000
```

---

## API Reference

### `POST /query`

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "best graphics card for gaming"}'
```

Cache miss response:
```json
{
  "query": "best graphics card for gaming",
  "cache_hit": false,
  "matched_query": null,
  "similarity_score": null,
  "result": "Top 5 semantic matches:\n\n1. [comp.sys.ibm.pc.hardware] (similarity: 0.612)\n   ...",
  "dominant_cluster": 2
}
```

Cache hit response (rephrased query):
```json
{
  "query": "top GPU for PC games",
  "cache_hit": true,
  "matched_query": "best graphics card for gaming",
  "similarity_score": 0.9134,
  "result": "Top 5 semantic matches:\n\n1. [comp.sys.ibm.pc.hardware] ...",
  "dominant_cluster": 2
}
```

### `GET /cache/stats`

```json
{
  "total_entries": 42,
  "hit_count": 17,
  "miss_count": 25,
  "hit_rate": 0.405,
  "avg_lookup_ms": 0.38
}
```

### `DELETE /cache`

Flushes all cached entries and resets stats to zero.

---

## Frontend Features

- **Search** with result cards showing matched documents, categories, and similarity scores
- **UI / JSON toggle** on search results — verify the raw `POST /query` response schema
- **UI / JSON toggle** on cache stats — verify the raw `GET /cache/stats` payload
- **Threshold slider** — adjust cache strictness from permissive (0.70) to strict (0.95)
- **Flush button** — calls `DELETE /cache` and resets stats live
- **Per-query metadata** — shows response time (ms) and dominant cluster ID

---

## Cluster Boundary Analysis

`setup.py` prints a semantic report after clustering:

1. **Core docs** — membership > 85% in a single cluster (clear topic ownership)
2. **Boundary docs** — split between two clusters (e.g., politics + firearms discussion)
3. **Uncertain docs** — near-uniform distributions (genuine model ambiguity)

This directly addresses: *"Show what lives in them, show what sits at their boundaries, and show where the model is genuinely uncertain."*
