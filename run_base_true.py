import json
from pipeline import prs_evaluator as P

cfg = json.load(open('examples/usecase4_bedrock_userguide/config_aug_P.json'))
faqs = json.load(open('examples/usecase4_bedrock_userguide/eval_heldout_paraphrase.json'))
# Force base model: pass lora_checkpoint=None explicitly, bypassing the CLI's
# version.json checkpoint_path fallback that silently loaded the P adapter.
prs = P.evaluate(faqs, cfg, lora_checkpoint=None)
print("BASE_TRUE PRS:", prs)
