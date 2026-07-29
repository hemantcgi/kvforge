"""Generate two matched-exposure UC4 training arms + disjointness audit vs held-out.
Arm D (dupes): each original question x N copies.
Arm P (paraphrases): N diverse paraphrases per concept, base-model generated,
                     filtered to be disjoint from the held-out paraphrase set.
Run on the GPU box (needs the base model)."""
import json, re, random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

N = 10            # exposure per concept (both arms matched)
JACCARD_CEIL = 0.6  # drop a train paraphrase too close to original or held-out
BASE = "meta-llama/Llama-3.2-3B-Instruct"
D = "examples/usecase4_bedrock_userguide"

def toks(s):
    return set(re.sub(r"[^\w\s]", " ", s.lower()).split())
def jac(a, b):
    A, B = toks(a), toks(b)
    return len(A & B) / max(1, len(A | B))

train = json.load(open(f"{D}/faqs.json"))
train = train if isinstance(train, list) else train.get("faqs", [])
held = json.load(open(f"{D}/eval_heldout_paraphrase.json"))
def strip(s): return re.sub(r"\s*\(variant\s+\d+\)\s*$", "", s.strip())

# align held-out to concept by answer text
held_by_ans = {h["answer"].strip(): h["question"] for h in held}
concepts = []
for x in train:
    q = strip(x.get("question", "")); a = x.get("answer", "")
    concepts.append({"q": q, "a": a, "held": held_by_ans.get(a.strip(), "")})
assert len(concepts) == 50, len(concepts)

tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="cuda").eval()

def gen_paraphrases(q, want):
    out = []
    for temp in (0.8, 1.0):  # two passes for variety
        prompt = tok.apply_chat_template(
            [{"role": "user", "content":
              f"Rewrite this question in {want} different ways with the SAME meaning but "
              f"varied wording and structure. Output ONLY the rewritten questions, one per line, no numbering.\n\nQuestion: {q}"}],
            add_generation_prompt=True, tokenize=False)
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
        with torch.no_grad():
            o = model.generate(**ids, max_new_tokens=400, do_sample=True, temperature=temp,
                               top_p=0.95, pad_token_id=tok.eos_token_id)
        txt = tok.decode(o[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        for line in txt.splitlines():
            line = re.sub(r"^\s*[\d\-\.\)\*]+\s*", "", line).strip().strip('"')
            if line.endswith("?") and 3 <= len(line.split()) <= 40:
                out.append(line)
    return out

random.seed(0)
armP, armD, audit = [], [], []
for i, c in enumerate(concepts):
    cands = gen_paraphrases(c["q"], 16)
    kept, seen = [], set()
    for p in cands:
        key = p.lower()
        if key in seen: continue
        if jac(p, c["q"]) > JACCARD_CEIL: continue          # too close to original
        if c["held"] and jac(p, c["held"]) > JACCARD_CEIL: continue  # too close to held-out
        seen.add(key); kept.append(p)
        if len(kept) >= N: break
    # disjointness audit: max Jaccard of kept paraphrases vs this concept's held-out question
    maxj = max((jac(p, c["held"]) for p in kept), default=0.0) if c["held"] else 0.0
    audit.append({"i": i, "n_kept": len(kept), "max_heldout_jac": round(maxj, 3)})
    for p in kept:
        armP.append({"question": p, "answer": c["a"]})
    for k in range(N):
        armD.append({"question": f'{c["q"]} (variant {k})', "answer": c["a"]})
    print(f"[{i+1}/50] kept {len(kept)} paraphrases, max held-out Jaccard {maxj:.2f}")

json.dump(armP, open(f"{D}/faqs_aug_paraphrase.json", "w"), indent=2)
json.dump(armD, open(f"{D}/faqs_aug_dupes.json", "w"), indent=2)
ns = [a["n_kept"] for a in audit]; mj = [a["max_heldout_jac"] for a in audit]
print("=== AUDIT ===")
print(f"Arm P rows: {len(armP)}  Arm D rows: {len(armD)}")
print(f"paraphrases/concept: min {min(ns)} mean {sum(ns)/len(ns):.1f} max {max(ns)}")
print(f"train<->heldout max-Jaccard per concept: overall max {max(mj):.3f} mean {sum(mj)/len(mj):.3f}")
print(f"concepts with any train paraphrase >0.5 Jaccard to held-out: {sum(1 for x in mj if x>0.5)}/50")
json.dump(audit, open(f"{D}/aug_disjointness_audit.json", "w"), indent=2)
