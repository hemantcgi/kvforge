#!/usr/bin/env bash
# R4: 4-chunk rerotation investigation + E5: Enhanced Tier latency
set -e
cd ~/kvforge
source venv/bin/activate
export CUDA_VISIBLE_DEVICES=0
python3 -u -c "
import torch, os, sys, time, tempfile, numpy as np
os.environ['HF_HOME'] = os.path.expanduser('~/.cache/huggingface')
sys.path.insert(0, os.path.expanduser('~/kvforge'))
from transformers import AutoModelForMultimodalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from core.kv_utils import compute_per_token_kv_as_list, save_token_kv_list, load_token_kv_list
from pipeline.kv_inference import _rerotate_fulltoken_chunks

model = AutoModelForMultimodalLM.from_pretrained('google/gemma-4-E2B-it', dtype=torch.float16, device_map='cuda:0', low_cpu_mem_usage=True)
tokenizer = AutoTokenizer.from_pretrained('google/gemma-4-E2B-it')
lm = model.model.language_model

def cap(text):
    inp = tokenizer(text, return_tensors='pt').to('cuda:0')
    with torch.no_grad(): out = lm(**inp, use_cache=True)
    return compute_per_token_kv_as_list(out.past_key_values), inp['input_ids'].shape[1]

def gen(all_kvs, label):
    cl, ql = all_kvs[0][0].shape[2], 0
    q = 'Question: How tall is the Eiffel Tower?\nAnswer:'
    inp = tokenizer(q, return_tensors='pt').to('cuda:0')
    ql = inp['input_ids'].shape[1]
    attn = torch.ones(1, cl + ql, device='cuda:0', dtype=torch.long)
    pos = torch.arange(cl, cl + ql, dtype=torch.long, device='cuda:0').unsqueeze(0)
    t0 = time.time()
    pkv = DynamicCache(config=model.config, ddp_cache_data=[(k.to('cuda:0').contiguous(), v.to('cuda:0').contiguous()) for k, v in all_kvs])
    with torch.no_grad():
        out = model.generate(input_ids=inp['input_ids'], attention_mask=attn, position_ids=pos, past_key_values=pkv, max_new_tokens=50, do_sample=False, repetition_penalty=1.3, no_repeat_ngram_size=4)
    ans = tokenizer.decode(out[0][ql:], skip_special_tokens=True)
    ok = '330' in ans or 'meters' in ans
    print('  %s: %.1fs | %s | %s' % (label, time.time()-t0, 'PASS' if ok else 'FAIL', repr(ans[:120])))

c1 = 'The Eiffel Tower is located in Paris, France. It was built in 1889 by Gustave Eiffel.'
c2 = 'The tower is 330 meters tall and is one of the most recognizable structures in the world.'
c3 = 'It was the tallest structure in the world until 1930 when the Chrysler Building was built.'
c4 = 'Today it remains one of the most visited monuments in the world.'

kv1, l1 = cap(c1); kv2, l2 = cap(c2); kv3, l3 = cap(c3); kv4, l4 = cap(c4)
print('Chunks: %d, %d, %d, %d' % (l1, l2, l3, l4))

# 2-chunk with rerot
rerot2 = _rerotate_fulltoken_chunks([kv1, kv2], [l1], lm)
ak2 = []; [ak2.append((torch.cat([torch.from_numpy(c[i][0].astype(np.float16)).unsqueeze(0) for c in rerot2], dim=2), torch.cat([torch.from_numpy(c[i][1].astype(np.float16)).unsqueeze(0) for c in rerot2], dim=2))) for i in range(15)]
gen(ak2, '2-chunk rerot')

# 4-chunk with rerot
rerot4 = _rerotate_fulltoken_chunks([kv1, kv2, kv3, kv4], [l1, l1+l2, l1+l2+l3], lm)
ak4 = []; [ak4.append((torch.cat([torch.from_numpy(c[i][0].astype(np.float16)).unsqueeze(0) for c in rerot4], dim=2), torch.cat([torch.from_numpy(c[i][1].astype(np.float16)).unsqueeze(0) for c in rerot4], dim=2))) for i in range(15)]
gen(ak4, '4-chunk rerot')

# 4-chunk with NO rerot
rerot4n = _rerotate_fulltoken_chunks([kv1, kv2, kv3, kv4], [0, 0, 0], lm)
ak4n = []; [ak4n.append((torch.cat([torch.from_numpy(c[i][0].astype(np.float16)).unsqueeze(0) for c in rerot4n], dim=2), torch.cat([torch.from_numpy(c[i][1].astype(np.float16)).unsqueeze(0) for c in rerot4n], dim=2))) for i in range(15)]
gen(ak4n, '4-chunk no rerot')

# 4-chunk encoded at correct positions
def cap_at(text, offset):
    inp = tokenizer(text, return_tensors='pt').to('cuda:0')
    pid = torch.arange(offset, offset + inp['input_ids'].shape[1], device='cuda:0').unsqueeze(0)
    with torch.no_grad(): out = lm(input_ids=inp['input_ids'], position_ids=pid, use_cache=True)
    return compute_per_token_kv_as_list(out.past_key_values)

kv1c = cap_at(c1, 0); kv2c = cap_at(c2, l1); kv3c = cap_at(c3, l1+l2); kv4c = cap_at(c4, l1+l2+l3)
ak4c = []; [ak4c.append((torch.cat([torch.from_numpy(x[i][0].astype(np.float16)).unsqueeze(0) for x in [kv1c,kv2c,kv3c,kv4c]], dim=2), torch.cat([torch.from_numpy(x[i][1].astype(np.float16)).unsqueeze(0) for x in [kv1c,kv2c,kv3c,kv4c]], dim=2))) for i in range(15)]
gen(ak4c, '4-chunk encode-at-pos')

# text_rag baseline
t0 = time.time()
full = c1 + ' ' + c2 + ' ' + c3 + ' ' + c4
text = tokenizer.apply_chat_template([{'role':'user','content': full + '\n\nHow tall is the Eiffel Tower?'}], tokenize=False, add_generation_prompt=True)
ti = tokenizer(text, return_tensors='pt').to('cuda:0')
with torch.no_grad(): to = model.generate(**ti, max_new_tokens=50, do_sample=False)
tans = tokenizer.decode(to[0][ti['input_ids'].shape[1]:], skip_special_tokens=True)
print('  text_rag: %.1fs | %s' % (time.time()-t0, repr(tans[:120])))

# E5: Enhanced Tier latency
p = tempfile.NamedTemporaryFile(suffix='.npz', delete=False).name
t0 = time.time(); save_token_kv_list(kv1, p); st = time.time() - t0
t0 = time.time(); ld = load_token_kv_list(p); lt = time.time() - t0
os.unlink(p)
print('E5: save=%.3fs load=%.3fs (%d tokens)' % (st, lt, l1))
print('Done')
"
