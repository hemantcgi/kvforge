#!/usr/bin/env bash
# Rerotation + Enhanced Tier validation on UC4 Bedrock
# Runs on GPU1
set -e
cd ~/kvforge
source venv/bin/activate
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=1

echo "========================================"
echo " Rerotation + Enhanced Tier Validation"
echo " GPU 1"
echo "========================================"

# ── R1: Single-chunk baseline ────────────────────────────────────────────
echo ""
echo "[R1] Single-chunk kv_fulltoken baseline"
python3 -u tools/poc_kv_fulltoken_gemma4.py 2>&1 | tail -10

# ── R2-R4: Rerotation scenarios ──────────────────────────────────────────
echo ""
echo "[R2-R4] Rerotation multi-chunk accuracy"
python3 -u -c "
import torch, os, sys, numpy as np
os.environ['HF_HOME'] = os.path.expanduser('~/.cache/huggingface')
sys.path.insert(0, os.path.expanduser('~/kvforge'))
from transformers import AutoModelForMultimodalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from core.kv_utils import compute_per_token_kv_as_list
from pipeline.kv_inference import _rerotate_fulltoken_chunks

model = AutoModelForMultimodalLM.from_pretrained('google/gemma-4-E2B-it', dtype=torch.float16, device_map='cuda:0', low_cpu_mem_usage=True)
tokenizer = AutoTokenizer.from_pretrained('google/gemma-4-E2B-it')
lm = model.model.language_model

def cap(text):
    inp = tokenizer(text, return_tensors='pt').to('cuda:0')
    with torch.no_grad():
        out = lm(**inp, use_cache=True)
    return compute_per_token_kv_as_list(out.past_key_values), inp['input_ids'].shape[1]

def inject_and_generate(kv_list_list, offsets, label):
    rerot = _rerotate_fulltoken_chunks(kv_list_list, offsets, model)
    all_kvs = []
    for i in range(15):
        ks, vs = [], []
        for c in rerot:
            a = torch.from_numpy(c[i].astype(np.float16))
            ks.append(a[0].unsqueeze(0))
            vs.append(a[1].unsqueeze(0))
        all_kvs.append((torch.cat(ks, dim=2), torch.cat(vs, dim=2)))
    cl = all_kvs[0][0].shape[2]
    q = 'Question: How tall is the Eiffel Tower?\\nAnswer:'
    inp = tokenizer(q, return_tensors='pt').to('cuda:0')
    ql = inp['input_ids'].shape[1]
    attn = torch.ones(1, cl + ql, device='cuda:0', dtype=torch.long)
    pos = torch.arange(cl, cl + ql, dtype=torch.long, device='cuda:0').unsqueeze(0)
    pkv = DynamicCache(config=model.config, ddp_cache_data=[(k.to('cuda:0'), v.to('cuda:0')) for k, v in all_kvs])
    with torch.no_grad():
        out = model.generate(input_ids=inp['input_ids'], attention_mask=attn, position_ids=pos, past_key_values=pkv, max_new_tokens=50, do_sample=False, repetition_penalty=1.3, no_repeat_ngram_size=4)
    ans = tokenizer.decode(out[0][ql:], skip_special_tokens=True)
    ok = '330' in ans or 'meters' in ans
    print(f'  {label}: {repr(ans[:100])} {\"PASS\" if ok else \"FAIL\"}')

c1 = 'The Eiffel Tower is located in Paris, France. It was built in 1889 by Gustave Eiffel.'
c2 = 'The tower is 330 meters tall and is one of the most recognizable structures in the world.'
c3 = 'It was the tallest structure in the world until 1930 when the Chrysler Building was built.'
c4 = 'Today it remains one of the most visited monuments in the world.'

kv1, l1 = cap(c1)
kv2, l2 = cap(c2)
kv3, l3 = cap(c3)
kv4, l4 = cap(c4)
print(f'Chunk lengths: {l1}, {l2}, {l3}, {l4}')

inject_and_generate([kv1], [], 'R2: 1 chunk')
inject_and_generate([kv1, kv2], [l1], 'R3: 2 chunks, rerot')
inject_and_generate([kv1, kv2, kv3, kv4], [l1, l1+l2, l1+l2+l3], 'R4: 4 chunks, rerot')
# Control: both at pos 0
inject_and_generate([kv1, kv2], [0], 'R2-control: 2 chunks, no rerot')
" 2>&1

# ── R6: Repetition penalty sweep ─────────────────────────────────────────
echo ""
echo "[R6] Repetition penalty sweep on kv_fulltoken"
for rp in 1.0 1.1 1.2 1.3 1.5; do
    echo "  rep_penalty=$rp"
    python3 -u -c "
import torch, os, sys
os.environ['HF_HOME'] = os.path.expanduser('~/.cache/huggingface')
sys.path.insert(0, os.path.expanduser('~/kvforge'))
from transformers import AutoModelForMultimodalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

model = AutoModelForMultimodalLM.from_pretrained('google/gemma-4-E2B-it', dtype=torch.float16, device_map='cuda:0', low_cpu_mem_usage=True)
tokenizer = AutoTokenizer.from_pretrained('google/gemma-4-E2B-it')
lm = model.model.language_model

ctx = 'The Eiffel Tower is located in Paris, France. It was built in 1889 by Gustave Eiffel. The tower is 330 meters tall and is one of the most recognizable structures in the world.'
ci = tokenizer(ctx, return_tensors='pt').to('cuda:0')
with torch.no_grad():
    co = lm(**ci, use_cache=True)
all_kvs = [(l.keys, l.values) for l in co.past_key_values.layers]
cl = all_kvs[0][0].shape[2]
q = 'Question: How tall is the Eiffel Tower?\\nAnswer:'
inp = tokenizer(q, return_tensors='pt').to('cuda:0')
ql = inp['input_ids'].shape[1]
attn = torch.ones(1, cl + ql, device='cuda:0', dtype=torch.long)
pos = torch.arange(cl, cl + ql, dtype=torch.long, device='cuda:0').unsqueeze(0)
pkv = DynamicCache(config=model.config, ddp_cache_data=[(k.to('cuda:0'), v.to('cuda:0')) for k, v in all_kvs])
with torch.no_grad():
    out = model.generate(input_ids=inp['input_ids'], attention_mask=attn, position_ids=pos, past_key_values=pkv, max_new_tokens=50, do_sample=False, repetition_penalty=$rp, no_repeat_ngram_size=4)
ans = tokenizer.decode(out[0][ql:], skip_special_tokens=True)
has = '330' in ans or 'meters' in ans
print(f'    rep=$rp -> {repr(ans[:100])} OK={has}')
" 2>&1
done

# ── E1-E4: Enhanced Tier validation ───────────────────────────────────────
echo ""
echo "[E1-E4] Enhanced Tier on-disk validation"
python3 -u -c "
import torch, os, sys, tempfile, numpy as np
os.environ['HF_HOME'] = os.path.expanduser('~/.cache/huggingface')
sys.path.insert(0, os.path.expanduser('~/kvforge'))
from transformers import AutoModelForMultimodalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from core.kv_utils import compute_per_token_kv_as_list, save_token_kv_list, load_token_kv_list, compute_per_token_kv

model = AutoModelForMultimodalLM.from_pretrained('google/gemma-4-E2B-it', dtype=torch.float16, device_map='cuda:0', low_cpu_mem_usage=True)
tokenizer = AutoTokenizer.from_pretrained('google/gemma-4-E2B-it')
lm = model.model.language_model

ctx = 'The Eiffel Tower is located in Paris, France. It was built in 1889 by Gustave Eiffel. The tower is 330 meters tall and is one of the most recognizable structures in the world.'
ci = tokenizer(ctx, return_tensors='pt').to('cuda:0')
with torch.no_grad():
    co = lm(**ci, use_cache=True)

# E1: Round-trip
kv_list = compute_per_token_kv_as_list(co.past_key_values)
p = tempfile.NamedTemporaryFile(suffix='.npz', delete=False).name
save_token_kv_list(kv_list, p)
loaded = load_token_kv_list(p)
os.unlink(p)
e1_ok = len(loaded) == 15 and all(loaded[i].shape == kv_list[i].shape for i in range(15))
print(f'  [E1] Round-trip: {len(loaded)} layers, shapes match={e1_ok}')

# E3: Generate from loaded
all_kvs = []
for i in range(15):
    k = torch.from_numpy(loaded[i][0].astype(np.float16)).unsqueeze(0).to('cuda:0')
    v = torch.from_numpy(loaded[i][1].astype(np.float16)).unsqueeze(0).to('cuda:0')
    all_kvs.append((k, v))
cl = all_kvs[0][0].shape[2]
q = 'Question: How tall is the Eiffel Tower?\\nAnswer:'
inp = tokenizer(q, return_tensors='pt').to('cuda:0')
ql = inp['input_ids'].shape[1]
attn = torch.ones(1, cl + ql, device='cuda:0', dtype=torch.long)
pos = torch.arange(cl, cl + ql, dtype=torch.long, device='cuda:0').unsqueeze(0)
pkv = DynamicCache(config=model.config, ddp_cache_data=[(k,v) for k,v in all_kvs])
with torch.no_grad():
    out = model.generate(input_ids=inp['input_ids'], attention_mask=attn, position_ids=pos, past_key_values=pkv, max_new_tokens=50, do_sample=False, repetition_penalty=1.3, no_repeat_ngram_size=4)
ans = tokenizer.decode(out[0][ql:], skip_special_tokens=True)
e3_ok = '330' in ans or 'meters' in ans
print(f'  [E3] Generate from loaded: {repr(ans[:100])} OK={e3_ok}')

# E2: TurboQuant round-trip
from core.kv_utils import save_token_kv_list as sq, load_token_kv_list as lq
from addons.turboquant.config import TurboQuantConfig
tq = TurboQuantConfig(key_bits=3, value_bits=4)
p2 = tempfile.NamedTemporaryFile(suffix='.npz', delete=False).name
sq(kv_list, p2, tq_config=tq)
loaded_tq = lq(p2)
os.unlink(p2)
e2_ok = len(loaded_tq) == 15 and all(loaded_tq[i].shape == kv_list[i].shape for i in range(15))
print(f'  [E2] TurboQuant round-trip: {len(loaded_tq)} layers, shapes match={e2_ok}')

print(f'  ALL ENHANCED TIER: {\"PASS\" if e1_ok and e2_ok and e3_ok else \"FAIL\"}')
" 2>&1

echo ""
echo "========================================"
echo " Rerotation + Enhanced Tier validation complete"
echo "========================================"
