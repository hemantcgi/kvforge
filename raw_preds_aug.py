import json
import core.model_loader as model_loader
from transformers import pipeline as hf_pipeline
from pipeline import prs_evaluator as P
from eval import metrics as M

ho = json.load(open('examples/usecase4_bedrock_userguide/eval_heldout_paraphrase.json'))
for tag in ['D', 'P']:
    cfg = json.load(open(f'examples/usecase4_bedrock_userguide/config_aug_{tag}.json'))
    model_loader.init(cfg)
    model, tok = model_loader.load(f'examples/usecase4_bedrock_userguide/lora_ckpt_aug_{tag}/v1/')
    pipe = hf_pipeline('text-generation', model=model, tokenizer=tok, max_new_tokens=256, do_sample=False)
    print('##### ARM', tag, '#####')
    for x in ho[:3]:
        q = x['question']; gt = x['answer']
        ans = P._generate_parametric(q, pipe, tok, 'chat')
        print('Q:', q)
        print('  EM=', M.exact_match(ans, gt), 'F1=%.3f' % M.token_f1(ans, gt))
        print('  GOLD:', gt[:150])
        print('  PRED:', ans[:150])
        print()
