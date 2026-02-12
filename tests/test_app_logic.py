import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.app import create_annotation

def test_create_annotation():
    result = {
        'source_id': '123',
        'text_en': 'Hello World',
        'group': 'Test Group',
        'name': 'Test Name'
    }
    matched_names = ['kw1']
    kept_ids = ['1']
    added_kws = ['New KW (ID: 2)']
    suggested_kws = ['kw3']
    final_new_kws = ['kw4']

    annotation = create_annotation(result, matched_names, kept_ids, added_kws, suggested_kws, final_new_kws)

    assert annotation['source_id'] == '123'
    assert annotation['text'] == 'Hello World'
    assert annotation['group'] == 'Test Group'
    assert annotation['name'] == 'Test Name'
    assert annotation['original_matched'] == ['kw1']
    assert annotation['kept_ids'] == ['1']
    assert annotation['added_existing_ids'] == ['2']
    assert annotation['accepted_new_keywords'] == ['kw4']

    print("create_annotation test passed!")

if __name__ == "__main__":
    test_create_annotation()
