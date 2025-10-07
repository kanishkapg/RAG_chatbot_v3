# Hybrid Search Algorithm Explanation

## What is Hybrid Search?

Hybrid search combines **two different search methods** to get better results than either method alone:

1. **Semantic Search** - Understands meaning and context (using AI embeddings)
2. **Lexical Search** - Finds exact keyword matches (using TF-IDF)

## How Does It Work? (Step by Step)

### Step 1: Dual Search
When you search for "maternity leave policy":
- **Semantic search** finds documents about parental benefits, time off, employee rights (meaning-based)
- **Lexical search** finds documents containing exact words "maternity", "leave", "policy" (keyword-based)
- Each method returns `top_k × 2` candidates (if you want 5 final results, each returns 10)

### Step 2: Score Normalization
Both methods produce different score ranges:
- Semantic scores might be 0.1 to 0.9
- Lexical scores might be 0.0 to 0.3

We normalize both to 0-1 range so they're comparable.

### Step 3: Score Fusion
For each document that appeared in either search:
```
final_score = (semantic_weight × semantic_score) + (lexical_weight × lexical_score)
```

Default weights: 70% semantic + 30% lexical

### Step 4: Final Ranking
- Rank ALL candidate documents by their combined score
- Return top_k results

## Real Example

Query: "maternity leave policy"

**Semantic Search finds:**
- Doc A: 0.85 (high semantic relevance)
- Doc B: 0.70 
- Doc C: 0.60

**Lexical Search finds:**
- Doc A: 0.90 (contains exact keywords)
- Doc B: 0.20 (few keyword matches)
- Doc D: 0.80 (many keyword matches but different meaning)

**After normalization and fusion:**
- Doc A: 0.70×0.85 + 0.30×0.90 = 0.865 ← **Winner!**
- Doc B: 0.70×0.70 + 0.30×0.20 = 0.550
- Doc C: 0.70×0.60 + 0.30×0.00 = 0.420
- Doc D: 0.70×0.00 + 0.30×0.80 = 0.240

## Why Hybrid is Better

- **Semantic-only**: Might miss documents with exact keywords but different phrasing
- **Lexical-only**: Might miss documents with same meaning but different words
- **Hybrid**: Gets the best of both worlds!

## Key Parameters

- `semantic_weight` (default 0.7): How much to trust meaning-based search
- `lexical_weight` (default 0.3): How much to trust keyword-based search  
- `top_k` (default 5): Number of final results to return

## Your Original Understanding vs Reality

❌ **You thought**: "Takes top 5 from each method and combines them"
✅ **Reality**: "Takes candidates from both methods, scores ALL of them with fusion, then picks top 5"

The key difference: It considers EVERY document that appeared in either search, not just the top results from each.
