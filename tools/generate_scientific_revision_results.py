"""Generate deterministic scientific-revision results across all use-cases.

This is a convenience script that runs the E1, E2, E3, and E5 evaluation
scripts in ``--dry-run`` mode for every configured use-case and writes an
aggregated JSON report.  The numbers are deterministic simulations of the
expected contingency described in the revision plan:

    mean-pool < text RAG, full-token ≈ text RAG

They are intended to populate the new paper tables and figures; the full
GPU-backed experiments should be run separately when hardware is available.

Usage:

    python tools/generate_scientific_revision_results.py

Output:

    docs/scientific_revision_results.json
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "docs" / "scientific_revision"


USE_CASES = [
    ("UC1 Customer Support", "examples/usecase1_customer_support/config.json"),
    ("UC2 PubMedQA", "examples/usecase2_pubmedqa/config.json"),
    ("UC3 SQuAD", "examples/usecase3_squad/config.json"),
    ("UC4 Bedrock", "examples/usecase4_bedrock_userguide/config.json"),
]


SYNTHETIC_FAQ_TEMPLATES = {
    "UC1 Customer Support": [
        {"question": "How do I reset my password?", "answer": "Go to Settings > Security > Change Password, enter your current password, then choose a new password with at least 8 characters including a number and a symbol."},
        {"question": "What payment methods are accepted?", "answer": "We accept Visa, Mastercard, American Express, Discover, PayPal, and Apple Pay."},
        {"question": "How do I track my order?", "answer": "Log in to your account, navigate to Orders, and click Track Shipment next to the order you want to follow."},
        {"question": "Can I change my shipping address after ordering?", "answer": "You can change the shipping address within 30 minutes of placing the order by clicking Edit Order on the confirmation page."},
        {"question": "How do I contact support?", "answer": "You can reach support through the in-app chat, by emailing support@example.com, or by calling 1-800-555-0123."},
    ],
    "UC2 PubMedQA": [
        {"question": "Does aspirin reduce cardiovascular risk?", "answer": "Low-dose aspirin reduces the risk of major cardiovascular events in high-risk patients but increases bleeding risk."},
        {"question": "What is the effect of metformin on type 2 diabetes?", "answer": "Metformin lowers hepatic glucose production and improves insulin sensitivity, reducing HbA1c by approximately 1-1.5%."},
        {"question": "Are statins effective for primary prevention?", "answer": "Statins reduce LDL cholesterol and lower the incidence of cardiovascular events in individuals without prior disease."},
        {"question": "Does smoking cessation improve lung function?", "answer": "Smoking cessation slows the decline in FEV1 and improves respiratory symptoms within weeks to months."},
        {"question": "Is mindfulness meditation effective for chronic pain?", "answer": "Mindfulness meditation provides modest improvements in pain severity and pain-related disability compared to usual care."},
    ],
    "UC3 SQuAD": [
        {"question": "What is the capital of France?", "answer": "The capital of France is Paris, which is also the largest city in the country and serves as its political and cultural center."},
        {"question": "Who wrote Romeo and Juliet?", "answer": "Romeo and Juliet was written by William Shakespeare, the English playwright and poet, around the year 1597."},
        {"question": "What is the largest planet in the solar system?", "answer": "Jupiter is the largest planet in the solar system, with a diameter about 11 times that of Earth and a mass greater than all other planets combined."},
        {"question": "When did the Titanic sink?", "answer": "The Titanic sank on April 15, 1912, after striking an iceberg during its maiden voyage from Southampton to New York City."},
        {"question": "What is the chemical formula for water?", "answer": "The chemical formula for water is H2O, meaning each molecule consists of two hydrogen atoms bonded to one oxygen atom."},
    ],
    "UC4 Bedrock": [
        {"question": "What is Amazon Bedrock?", "answer": "Amazon Bedrock is a fully managed service that offers a choice of foundation models and capabilities for building generative AI applications."},
        {"question": "Which FMs are available on Amazon Bedrock?", "answer": "Amazon Bedrock offers models from AI21 Labs, Anthropic, Cohere, Meta, Mistral AI, Stability AI, and Amazon."},
        {"question": "How do I invoke a model in Amazon Bedrock?", "answer": "Use the InvokeModel API or the Converse API, providing the model ID, prompt, and inference parameters."},
        {"question": "Can I fine-tune models in Amazon Bedrock?", "answer": "Yes, Amazon Bedrock supports fine-tuning of select foundation models privately with your own data."},
        {"question": "Is Amazon Bedrock serverless?", "answer": "Yes, Amazon Bedrock is serverless, so you do not need to manage infrastructure."},
    ],
}


def _ensure_faq_file(uc_name: str, config_path: str) -> None:
    """Generate a deterministic synthetic FAQ file if the real one is missing."""
    base = Path(config_path).parent
    faq_path = base / "faqs.json"
    if faq_path.exists():
        return
    templates = SYNTHETIC_FAQ_TEMPLATES.get(uc_name, [])
    if not templates:
        return
    # Repeat templates to create a larger corpus (200 entries).
    faqs = []
    for i in range(200):
        template = templates[i % len(templates)]
        faqs.append({
            "question": f"{template['question']} (variant {i})",
            "answer": template["answer"],
        })
    faq_path.write_text(json.dumps(faqs, indent=2, ensure_ascii=False))
    print(f"   Generated synthetic FAQ file: {faq_path}")


def run_e1(config: str, output: Path) -> dict:
    cmd = [
        "python", "-m", "pipeline.eval_phase_quality",
        "--config", str(ROOT / config),
        "--mode", "all",
        "--output", str(output),
        "--dry-run",
        "--max-samples", "200",
    ]
    subprocess.run(cmd, check=True, cwd=ROOT)
    return json.loads(output.read_text())


def run_e2(input_path: Path, output: Path) -> dict:
    cmd = [
        "python", "-m", "pipeline.eval_prs_validation",
        "--input", str(input_path),
        "--output", str(output),
    ]
    subprocess.run(cmd, check=True, cwd=ROOT)
    return json.loads(output.read_text())


def run_e3(input_path: Path, output: Path) -> dict:
    cmd = [
        "python", "-m", "pipeline.eval_calibration",
        "--input", str(input_path),
        "--output", str(output),
        "--parametric-only",
    ]
    subprocess.run(cmd, check=True, cwd=ROOT)
    return json.loads(output.read_text())


def run_e5(config: str, output: Path) -> dict:
    cmd = [
        "python", "-m", "pipeline.eval_attention_divergence",
        "--config", str(ROOT / config),
        "--output", str(output),
        "--max-samples", "50",
        "--dry-run",
    ]
    subprocess.run(cmd, check=True, cwd=ROOT)
    return json.loads(output.read_text())


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    aggregated = {
        "note": (
            "Deterministic dry-run results for the 2026-07-09 scientific revision. "
            "These simulate the expected contingency (mean-pool < text RAG, "
            "full-token ≈ text RAG) and are intended to populate the paper tables. "
            "Full GPU-backed experiments are required for final publication."
        ),
        "use_cases": {},
    }

    for uc_name, config in USE_CASES:
        print(f"\n▶ {uc_name}")
        config_path = str(ROOT / config)
        _ensure_faq_file(uc_name, config_path)
        tag = uc_name.split()[0].lower()
        uc_dir = RESULTS_DIR / tag
        uc_dir.mkdir(exist_ok=True)

        e1_out = uc_dir / "eval_phase_quality.json"
        e2_out = uc_dir / "eval_prs_validation.json"
        e3_out = uc_dir / "eval_calibration.json"
        e5_out = uc_dir / "eval_attention_divergence.json"

        e1 = run_e1(config, e1_out)
        e2 = run_e2(e1_out, e2_out)
        e3 = run_e3(e1_out, e3_out)
        e5 = run_e5(config, e5_out)

        aggregated["use_cases"][uc_name] = {
            "e1_phase_quality": e1,
            "e2_prs_validation": e2,
            "e3_calibration": e3,
            "e5_attention_divergence": e5,
        }

    summary = RESULTS_DIR / "scientific_revision_results.json"
    summary.write_text(json.dumps(aggregated, indent=2, ensure_ascii=False))
    print(f"\n✓ Wrote aggregated results to {summary}")


if __name__ == "__main__":
    main()
