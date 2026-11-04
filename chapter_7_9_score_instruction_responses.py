"""Use a local Ollama model to score instruction-tuned model responses."""

import json
import re
from pathlib import Path

from tqdm import tqdm

from chapter_7_2_prepare_instruction_data import format_input
from chapter_7_8_query_ollama import query_model


ROOT = Path(__file__).resolve().parent
INPUT_PATH = ROOT / "instruction-data-with-response.json"
OUTPUT_PATH = ROOT / "instruction-data-with-scores.json"
JUDGE_MODEL = "llama3"


def parse_score(text):
    """Extract an integer score from the judge response and validate its range."""
    match = re.search(r"(?<!\d)(100|[1-9]?\d)(?!\d)", text)
    if not match:
        raise ValueError(f"No integer score found in judge response: {text!r}")
    score = int(match.group(1))
    if not 0 <= score <= 100:
        raise ValueError(f"Score is outside 0-100: {score}")
    return score


def generate_model_scores(json_data, json_key="model_response", model=JUDGE_MODEL):
    """Score each generated answer against its reference output."""
    scores = []
    for index, entry in enumerate(tqdm(json_data, desc=f"Scoring with {model}")):
        if json_key not in entry:
            print(f"Skipping entry {index}: missing key {json_key!r}")
            entry["llama_score"] = None
            entry["llama_score_raw"] = None
            scores.append(None)
            continue

        prompt = (
            "You are an impartial evaluator. Compare the candidate response with "
            "the reference answer, considering correctness, relevance, and whether "
            "it follows the instruction. Give a score from 0 to 100, where 100 is "
            "fully correct. Return only one integer, with no explanation.\n\n"
            f"<input>\n{format_input(entry)}\n</input>\n\n"
            f"<reference_answer>\n{entry['output']}\n</reference_answer>\n\n"
            f"<candidate_response>\n{entry[json_key]}\n</candidate_response>"
        )
        raw_score = query_model(prompt, model, print_response_details=False).strip()
        try:
            score = parse_score(raw_score)
        except ValueError as error:
            print(f"\nCould not parse score for entry {index}: {error}")
            score = None

        entry["llama_score"] = score
        entry["llama_score_raw"] = raw_score
        scores.append(score)

        if index < 3:
            print(f"\nExample {index + 1}")
            print("Dataset response:\n>>", entry["output"])
            print("\nModel response:\n>>", entry[json_key])
            print("\nLlama score:\n>>", raw_score)
            print("Parsed score:", score)
            print("-" * 30)

    return scores


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"{INPUT_PATH} not found. Run chapter_7_3_instruction_finetuning.py first."
        )

    test_data = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    scores = generate_model_scores(test_data)
    valid_scores = [score for score in scores if score is not None]

    OUTPUT_PATH.write_text(
        json.dumps(test_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nScored {len(valid_scores)}/{len(test_data)} responses.")
    if valid_scores:
        print(f"Mean Llama score: {sum(valid_scores) / len(valid_scores):.1f}/100")
    print(f"Saved scores to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
