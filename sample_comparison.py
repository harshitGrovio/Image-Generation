#!/usr/bin/env python3
"""Show sample comparison for a few queries"""

import json
import requests

queries_to_sample = ['instagram-post', 'product-listings', 'edit-images', 'youtube-thumbnails']

print("="*70)
print("SAMPLE QUERY COMPARISONS")
print("="*70)

with open('query_responses.json', 'r') as f:
    expected = json.load(f)

for query in queries_to_sample:
    print(f"\n{query.upper()}")
    print("-" * 70)
    
    # Get expected
    exp = expected[query]
    
    # Get actual
    resp = requests.get(f'http://localhost:5005/models?query={query}')
    act = resp.json()
    
    print(f"  Query:                 {act.get('query')}")
    print(f"  Total:                 {act.get('total')}")
    print(f"  Models count:          {len(act.get('models', []))}")
    
    has_editing = 'editing_models' in act
    print(f"  Has editing_models:    {has_editing}")
    if has_editing:
        print(f"  Editing models count:  {len(act.get('editing_models', []))}")
    
    print(f"  Has system_prompt:     {'system_prompt' in act}")
    print(f"  Has sample_prompt:     {'sample_prompt' in act}")
    print(f"  Has prompt_tips:       {'prompt_tips' in act}")
    print(f"  Has platform_sizes:    {'platform_sizes' in act}")
    
    # Show first 3 model IDs
    if act.get('models'):
        model_ids = [m.get('model') for m in act['models'][:3]]
        print(f"  First 3 models:        {', '.join(model_ids)}")

print("\n" + "="*70)
