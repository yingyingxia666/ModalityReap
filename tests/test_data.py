import json

from modality_reap.data import build_dataset_specs, discover_jsonl_files, normalize_messages


def test_build_dataset_specs_remaps_root():
    specs = build_dataset_specs('/tmp/datasets')
    assert str(specs[0].path).startswith('/tmp/datasets/')


def test_discover_jsonl_files_finds_single_file(tmp_path):
    path = tmp_path / 'sample.jsonl'
    path.write_text('{"messages": []}\n', encoding='utf-8')
    files = discover_jsonl_files(str(path))
    assert files == [path]


def test_normalize_messages_handles_multimodal_content():
    messages = [
        {
            'role': 'user',
            'content': [
                {'audio_path': 'a.wav'},
                {'text': 'hello'},
            ],
        }
    ]
    normalized = normalize_messages(messages)
    assert normalized == [{'role': 'user', 'content': '<audio>\nhello'}]
