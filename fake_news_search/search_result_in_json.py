import json
from retrieval_pipeline import retrieve_evidence_for_claim
from llm_verdict import compute_verdict_and_explanation

def main():
    claim = input("Enter a claim to retrieve evidence for:\n> ").strip()
    print(f"\nSearching evidence for: \"{claim}\"\n")

    evidence_list = retrieve_evidence_for_claim(claim, num_results=5)

    if not evidence_list:
        print(json.dumps({"claim": claim, "evidence": [], "message": "No results found."}, indent=2))
        return

    # Use full text as evidence, fallback to snippet if full text is missing
    processed_evidence = [
        {
            "title": ev.get("title", ""),
            "url": ev.get("url", ""),
            "evidence_text": ev.get("raw_text", "").strip() or ev.get("snippet", "").strip()
        }
        for ev in evidence_list[:5]
    ]

    verdict_result = compute_verdict_and_explanation(claim, processed_evidence)

    final_output = {
        "claim": claim,
        "verdict": verdict_result["verdict"],
        "explanation": verdict_result["explanation"],
        "evidence": processed_evidence
    }

    print(json.dumps(final_output, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
