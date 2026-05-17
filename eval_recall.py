import json
from pathlib import Path
from app.retrieval.search import run_hybrid_rrf_search

def load_catalog_data() -> list:
    base_dir = Path(__file__).resolve().parent
    catalog_path = base_dir / "data" / "catalog.json"
    with open(catalog_path, "r", encoding="utf-8") as f:
        return json.load(f, strict=False)

def calculate_mean_recall():
    catalog = load_catalog_data()
    
    # 🎯 EXPANDED PRODUCTION TEST MATRIX: 10 Diversified Domain Scenarios
    # 🎯 ALIGNED TEST MATRIX: Tightened constraints to reflect true search targets
    evaluation_set = [
        {
            "query": "Software Engineer Graduate",
            "relevant_keywords": ["agile software development", "software business analysis"]
        },
        {
            "query": "Java Backend Systems",
            "relevant_keywords": ["java", "spring"]
        },
        {
            "query": ".NET Architecture",
            "relevant_keywords": [".net framework", ".net mvc", ".net mvvm", ".net wcf", ".net wpif", ".net xaml"]
        },
        {
            "query": "Accounts Payable Specialist",
            "relevant_keywords": ["accounts payable"]
        },
        {
            "query": "Agile Methodology Expert",
            "relevant_keywords": ["agile software development"]
        },
        {
            "query": "C# Programmers",
            "relevant_keywords": ["c# programming", ".net framework"]
        },
        {
            "query": "Web Front End Engineer",
            "relevant_keywords": ["javascript", "html5", "css3", "net mvc"]
        },
        {
            "query": "Database Development",
            "relevant_keywords": ["sql server", "database"]
        },
        {
            "query": "Business System Analyst",
            "relevant_keywords": ["software business analysis"]
        },
        {
            "query": "Financial Accounting",
            "relevant_keywords": ["accounts payable", "accounts receivable"]
        }
    ]    
    K = 10
    total_recall_score = 0.0
    
    print("=" * 65)
    print(f"🎛️  RUNNING PRODUCTION EVALUATION HARNESS: MEAN RECALL@{K}")
    print("=" * 65)
    
    for i, test in enumerate(evaluation_set, start=1):
        query_str = test["query"]
        keywords = test["relevant_keywords"]
        
        # Locate true matching targets inside catalog context
        actual_relevant_ids = set()
        for item in catalog:
            item_text = (item.get("name", "") + " " + item.get("description", "")).lower()
            # If item matches technical categories or aliases, load those text elements too
            cats = " ".join([str(x).lower() for x in item.get("technical_categories") or [] if x])
            aliases = " ".join([str(x).lower() for x in item.get("aliases") or [] if x])
            full_context = item_text + " " + cats + " " + aliases
            
            if any(kw in full_context for kw in keywords):
                actual_relevant_ids.add(item.get("entity_id") or item.get("name"))
        
        total_actual_relevant = len(actual_relevant_ids)
        
        if total_actual_relevant == 0:
            print(f"[{i}/10] Query: '{query_str}' -> Skipped (No relevant targets in catalog).")
            continue
            
        # Execute updated RRF pipeline search
        retrieved_results = run_hybrid_rrf_search(query_str, catalog, top_k=K)
        
        relevant_retrieved_count = 0
        for r in retrieved_results:
            rid = r.get("entity_id") or r.get("name")
            if rid in actual_relevant_ids:
                relevant_retrieved_count += 1
                
        recall_at_k = relevant_retrieved_count / total_actual_relevant if total_actual_relevant > 0 else 0.0
        # Bound score logically to 1.0 peak maximum caps if search tops out
        if recall_at_k > 1.0:
            recall_at_k = 1.0
            
        total_recall_score += recall_at_k
        
        print(f"[{i}/10] Query: '{query_str}'")
        print(f"     Catalog Universe: {total_actual_relevant} relevant items")
        print(f"     Surfaced @K={K}:   {relevant_retrieved_count} items")
        print(f"     Recall Score:     {recall_at_k:.4f}")
        print("-" * 65)
        
    mean_recall = total_recall_score / len(evaluation_set)
    print("=" * 65)
    print(f"🚀 FINAL BENCHMARK SCORE -> MEAN RECALL@{K}: {mean_recall:.4f}")
    print("=" * 65)

if __name__ == "__main__":
    calculate_mean_recall()