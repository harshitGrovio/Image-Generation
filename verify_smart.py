#!/usr/bin/env python3
"""
Deep verification: Hit all queries and match against query_responses.json
Checks structure, field presence, counts, and data integrity
"""

import json
import requests
import sys
from typing import Dict, List, Any

# Load expected responses
print("Loading query_responses.json...")
with open('query_responses.json', 'r') as f:
    expected = json.load(f)

print(f"Found {len(expected)} queries to test\n")

def deep_compare(query: str, expected_data: Dict, actual_data: Dict) -> List[str]:
    """Deep comparison of expected vs actual response"""
    issues = []
    
    # 1. Check query field
    if actual_data.get('query') != query:
        issues.append(f"  query: expected '{query}', got '{actual_data.get('query')}'")
    
    # 2. Check description
    expected_desc = expected_data.get('description')
    actual_desc = actual_data.get('description')
    if expected_desc != actual_desc:
        issues.append(f"  description: mismatch")
    
    # 3. Check total count
    if expected_data.get('total') != actual_data.get('total'):
        issues.append(f"  total: expected {expected_data.get('total')}, got {actual_data.get('total')}")
    
    # 4. Check models array length
    expected_models = expected_data.get('models', [])
    actual_models = actual_data.get('models', [])
    if len(expected_models) != len(actual_models):
        issues.append(f"  models length: expected {len(expected_models)}, got {len(actual_models)}")
    
    # 5. Check models content - extract model IDs
    if isinstance(expected_models, list) and len(expected_models) > 0:
        if isinstance(expected_models[0], dict):
            expected_model_ids = [m.get('model') for m in expected_models]
            actual_model_ids = [m.get('model') for m in actual_models]
            
            # Check if all expected models are present
            for mid in expected_model_ids:
                if mid and mid not in actual_model_ids:
                    issues.append(f"  missing model: {mid}")
            
            # Check if there are unexpected models
            for mid in actual_model_ids:
                if mid and mid not in expected_model_ids:
                    issues.append(f"  unexpected model: {mid}")
    
    # 6. Check editing_models field presence and content
    expected_has_editing = 'editing_models' in expected_data
    actual_has_editing = 'editing_models' in actual_data
    
    if expected_has_editing != actual_has_editing:
        if expected_has_editing:
            issues.append(f"  MISSING 'editing_models' field (should be present)")
        else:
            issues.append(f"  UNEXPECTED 'editing_models' field (should NOT be present)")
    elif expected_has_editing:
        # Both have editing_models, compare them
        expected_editing = expected_data.get('editing_models', [])
        actual_editing = actual_data.get('editing_models', [])
        if set(expected_editing) != set(actual_editing):
            issues.append(f"  editing_models mismatch: expected {expected_editing}, got {actual_editing}")
    
    # 7. Check optional fields presence
    optional_fields = ['system_prompt', 'sample_prompt', 'prompt_tips', 
                      'platform_sizes', 'recommended_size', 'notes', 'available_queries']
    
    for field in optional_fields:
        expected_has = field in expected_data
        actual_has = field in actual_data
        
        if expected_has and not actual_has:
            issues.append(f"  MISSING field '{field}'")
        elif not expected_has and actual_has:
            # Only report if it's not available_queries (that's added by API)
            if field != 'available_queries':
                issues.append(f"  UNEXPECTED field '{field}' (should not be present)")
    
    return issues

# Test all queries
results = {
    'passed': [],
    'failed': [],
    'errors': []
}

print("="*70)
print("TESTING ALL QUERIES")
print("="*70)

for idx, query in enumerate(sorted(expected.keys()), 1):
    print(f"\n[{idx}/{len(expected)}] Testing: {query}")
    expected_data = expected[query]
    
    try:
        # Hit the API
        response = requests.get(f'http://localhost:5005/models?query={query}', timeout=10)
        response.raise_for_status()
        actual_data = response.json()
        
        # Deep compare
        issues = deep_compare(query, expected_data, actual_data)
        
        if issues:
            results['failed'].append({
                'query': query,
                'issues': issues
            })
            print(f"  ✗ FAILED")
            for issue in issues:
                print(f"    {issue}")
        else:
            results['passed'].append(query)
            print(f"  ✓ PASSED")
            print(f"    - Total: {actual_data.get('total')} models")
            print(f"    - Models: {len(actual_data.get('models', []))}")
            if 'editing_models' in actual_data:
                print(f"    - Editing models: {len(actual_data.get('editing_models', []))}")
            
    except Exception as e:
        results['errors'].append({
            'query': query,
            'error': str(e)
        })
        print(f"  ✗ ERROR: {str(e)}")

# Summary
print("\n" + "="*70)
print("VERIFICATION SUMMARY")
print("="*70)
print(f"Total queries tested: {len(expected)}")
print(f"✓ Passed: {len(results['passed'])}")
print(f"✗ Failed: {len(results['failed'])}")
print(f"⚠ Errors: {len(results['errors'])}")

if results['passed']:
    print(f"\n{'='*70}")
    print(f"PASSED QUERIES ({len(results['passed'])})")
    print(f"{'='*70}")
    for q in results['passed']:
        print(f"  ✓ {q}")

if results['failed']:
    print(f"\n{'='*70}")
    print(f"FAILED QUERIES ({len(results['failed'])})")
    print(f"{'='*70}")
    for item in results['failed']:
        print(f"\n  ✗ {item['query']}")
        for issue in item['issues']:
            print(f"    {issue}")

if results['errors']:
    print(f"\n{'='*70}")
    print(f"ERRORS ({len(results['errors'])})")
    print(f"{'='*70}")
    for item in results['errors']:
        print(f"\n  ⚠ {item['query']}: {item['error']}")

# Exit code
if results['failed'] or results['errors']:
    print("\n❌ VERIFICATION FAILED")
    sys.exit(1)
else:
    print("\n✅ ALL QUERIES MATCH PERFECTLY!")
    sys.exit(0)
