from modality_reap.eval import EvalSample, rouge_l_f1, select_generation_subset, token_f1


def test_token_f1_matches_identical_text():
    assert token_f1("hello world", "hello world") == 1.0


def test_rouge_l_f1_rewards_subsequence_overlap():
    score = rouge_l_f1("rain and thunder", "rain thunder")
    assert 0.7 < score < 1.0


def test_select_generation_subset_respects_per_dataset_limits():
    samples = [
        EvalSample("a1", "AudioCaps-test", "audio", [], [], "x", ["x"]),
        EvalSample("a2", "AudioCaps-test", "audio", [], [], "x", ["x"]),
        EvalSample("t1", "UltraChat", "text", [], [], "x", ["x"]),
        EvalSample("t2", "UltraChat", "text", [], [], "x", ["x"]),
    ]
    selected = select_generation_subset(samples, audio_per_dataset=1, text_per_dataset=1)
    assert [sample.sample_id for sample in selected] == ["a1", "t1"]
