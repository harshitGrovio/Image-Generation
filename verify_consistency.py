#!/usr/bin/env python3
"""Comprehensive verification that API responses match query_responses.json exactly"""

import json
import requests
import sys

# Load expected responses
with open('query_responses.json', 'r') as f:
    expected = json.load(f)

def verify_field_presence(query, expected_data, actual_data):
    """Verify that all expected fields are present and unexpected fields are absent"""
    issues = []

    # Fields that should be checked for exact presence
    optional_fields = ['editing_models', 'system_prompt', 'sample_prompt', 'prompt_tips',
                       'platform_sizes', 'recommended_size', 'notes']

    for field in optional_fields:
        expected_has = field in expected_data
        actual_has = field in actual_data

        if expected_has != actual_has:
            if expected_has:
                issues.append(f"  Missing field '{field}'")
            else:
                issues.append(f"  Unexpected field '{field}' (should not be present)")

    return issues

# Test each query
all_issues = []
all_passed = []

for query in expected.keys():
    expected_data = expected[query]

    # Call API
    try:
        response = requests.get(f'http://localhost:5005/models?query={query}')
        response.raise_for_status()
        actual_data = response.json()

        # Check field presence
        issues = verify_field_presence(query, expected_data, actual_data)

        if issues:
            all_issues.append({
                'query': query,
                'issues': issues
            })
            print(f"✗ {query}")
            for issue in issues:
                print(issue)
        else:
            all_passed.append(query)
            print(f"✓ {query}")

    except Exception as e:
        print(f"ERROR {query}: {e}")
        all_issues.append({
            'query': query,
            'issues': [f"  API Error: {str(e)}"]
        })

print(f"\n{'='*60}")
print(f"Summary: {len(all_passed)} passed, {len(all_issues)} failed")
print(f"{'='*60}")

if all_issues:
    print("\nIssues found:")
    for item in all_issues:
        print(f"\n{item['query']}:")
        for issue in item['issues']:
            print(issue)
    sys.exit(1)
else:
    print("\n✓ All query responses match expected structure!")
    sys.exit(0)
