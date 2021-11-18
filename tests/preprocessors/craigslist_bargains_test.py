import pytest
from datasets import load_dataset, set_caching_enabled

from src.preprocessors import craigslist_bargains


class TestCraigslistBargainsPreprocessor:

    def setup(self):
        set_caching_enabled(False)

    def teardown(self):
        set_caching_enabled(True)

    @pytest.mark.parametrize("add_speaker_prefix", [True, False], ids=["Speaker", "No Speaker"])
    def test_call(self, add_speaker_prefix):
        processor = craigslist_bargains.CraigslistBargainsPreprocessor(
            add_speaker_prefix=add_speaker_prefix
        )

        ds = load_dataset("craigslist_bargains", split='train[:5]')

        result_ds = ds.map(  # type:ignore
            processor,
            with_indices=True,
            remove_columns=ds.column_names
        )

        assert set(result_ds.column_names) == {
            "idx", "input_sequence", "label", "additional_inputs", "choices", "domain"
        }

        assert len(result_ds) == 5
        result_ds = result_ds.sort('idx')

        def add_idx(ex, _idx):
            ex['idx'] = _idx
            return ex

        ds = ds.map(
            add_idx,
            with_indices=True,
        ).sort('idx')

        for idx, (result, expected) in enumerate(zip(result_ds, ds)):
            assert result['idx'] == idx
            result_sequence = result['input_sequence'].split("\n\n")
            expected_sequence = [
                f"{('Buyer' if i % 2 == 0 else 'Seller') + ': ' if add_speaker_prefix else ''}{utt}"
                for i, utt in enumerate(ds[idx]['utterance'])
            ]

            assert len(result_sequence) == len(expected_sequence), "Sequence length is wrong"
            for j, (r, e) in enumerate(zip(result_sequence, expected_sequence)):
                assert r == e, f"result[{idx}]['input_sequence'][{j}]"
            expected_label = processor.label_to_int[processor._get_label(
                expected['dialogue_acts']['intent'],
                expected['dialogue_acts']['price'],
                expected['agent_info']['Target']
            )]
            assert result['label'] == expected_label
            assert result['additional_inputs'] == [expected['items']['Price'][0]]

    @pytest.mark.parametrize("dialogue_acts,targets,expected", [
        [[["", "", ""], [-1.0, 10, -1.0]], [1369.0, 2283.0], "UNKNOWN"],
        [[["intro", "inquiry", "accept"], [-1.0, 10, -1.0]], [5, 15], "NEITHER"],
        [[["intro", "inquiry", "reject"], [-1.0, 10, -1.0]], [5, 15], "REJECT"],
        [[["intro", "inquiry", "accept"], [-1.0, 11, -1.0]], [5, 15], "SELLER"],
        [[["intro", "inquiry", "accept"], [-1.0, 9, -1.0]], [5, 15], "BUYER"]
    ])
    def test_get_label(self, dialogue_acts, targets, expected):
        processor = craigslist_bargains.CraigslistBargainsPreprocessor()
        assert processor._get_label(dialogue_acts[0], dialogue_acts[1], targets) == expected
