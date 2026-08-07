#!/usr/bin/env python3
"""Generate FAQs from chunks.json using Gemini API directly (bypass Qdrant)."""

import json
import os
import sys
import time
import google.generativeai as genai

CHUNKS_FILE = sys.argv[1] if len(sys.argv) > 1 else "chunks.json"
OUTPUT = sys.argv[2] if len(sys.argv) > 2 else "faqs_500.json"
TARGET = int(sys.argv[3]) if len(sys.argv) > 3 else 500

genai.configure(api_key=os.environ["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-2.5-flash")

chunks = json.load(open(CHUNKS_FILE))
print(f"Loaded {len(chunks)} chunks")

faqs = []
for i, chunk in enumerate(chunks):
    if len(faqs) >= TARGET:
        break
    
    prompt = f"""Based on the following text, generate one question-answer pair.
Return JSON: {{"question": "...", "answer": "...", "chunk_id": "{chunk['chunk_id']}"}}

Text: {chunk['text'][:2000]}"""
    
    try:
        resp = model.generate_content(prompt)
        text = resp.text.strip()
        text = text.replace("```json", "").replace("```", "").strip()
        faq = json.loads(text)
        faq["source"] = chunk["chunk_id"]
        faqs.append(faq)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(chunks)} chunks processed, {len(faqs)} FAQs generated")
    except Exception as e:
        print(f"  Error on chunk {i}: {e}")
    
    time.sleep(0.5)

json.dump(faqs, open(OUTPUT, "w"), indent=2)
print(f"\nDone: {len(faqs)} FAQs written to {OUTPUT}")
