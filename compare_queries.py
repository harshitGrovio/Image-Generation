#!/usr/bin/env python3
"""Compare API query responses with expected responses from query_responses.json"""

import json
import requests
import sys

# Load expected responses
with open('query_responses.json', 'r') as f:
    expected = json.load(f)

# Test each query
mismatches = []
matches = []

for query in expected.keys():
    expected_total = expected[query]['total']
    expected_models = len(expected[query].get('models', []))
    expected_editing = len(expected[query].get('editing_models', []))

    # Call API
    try:
        response = requests.get(f'http://localhost:5005/models?query={query}')
        response.raise_for_status()
        actual = response.json()

        actual_total = actual.get('total', 0)
        actual_models = len(actual.get('models', []))
        actual_editing = len(actual.get('editing_models', []))

        if expected_total != actual_total:
            mismatches.append({
                'query': query,
                'field': 'total',
                'expected': expected_total,
                'actual': actual_total
            })

        if expected_models != actual_models:
            mismatches.append({
                'query': query,
                'field': 'models_count',
                'expected': expected_models,
                'actual': actual_models
            })

        if expected_editing != actual_editing:
            mismatches.append({
                'query': query,
                'field': 'editing_models_count',
                'expected': expected_editing,
                'actual': actual_editing
            })

        if expected_total == actual_total and expected_models == actual_models and expected_editing == actual_editing:
            matches.append(query)
            print(f"✓ {query}: total={actual_total}, models={actual_models}, editing={actual_editing}")
        else:
            print(f"✗ {query}: Expected total={expected_total}, models={expected_models}, editing={expected_editing}")
            print(f"  Got: total={actual_total}, models={actual_models}, editing={actual_editing}")

    except Exception as e:
        print(f"ERROR {query}: {e}")
        mismatches.append({
            'query': query,
            'field': 'error',
            'expected': 'success',
            'actual': str(e)
        })

print(f"\n{'='*60}")
print(f"Summary: {len(matches)} matches, {len(mismatches)} mismatches")
print(f"{'='*60}")

if mismatches:
    print("\nMismatches:")
    for m in mismatches:
        print(f"  {m['query']} - {m['field']}: expected={m['expected']}, actual={m['actual']}")
    sys.exit(1)
else:
    print("\n✓ All queries match!")
    sys.exit(0)
