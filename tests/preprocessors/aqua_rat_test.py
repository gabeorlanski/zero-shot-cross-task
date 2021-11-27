import pytest
from datasets import load_dataset, set_caching_enabled

from src.preprocessors import aqua_rat, TaskMode


class TestCraigslistBargainsPreprocessor:

    def setup(self):
        set_caching_enabled(False)

    def teardown(self):
        set_caching_enabled(True)

    @pytest.mark.parametrize("mode", list(TaskMode), ids=list(map(str, TaskMode)))
    def test_call(self, mode):
        processor = aqua_rat.AquaRatPreprocessor()
        processor.set_mode(mode)

        ds = load_dataset("aqua_rat", "raw", split='train[:5]')

        result_ds = ds.map(  # type:ignore
            processor,
            with_indices=True,
            remove_columns=ds.column_names
        )

        expected_columns = {
            "idx",
            "label",
            "choices",
            "domain",
            "choice_string"
        }
        if mode == TaskMode.QA:
            expected_columns.update(["question", "context"])
        elif mode == TaskMode.ENTAILMENT:
            expected_columns.update(['premise', 'hypothesis'])
        else:
            expected_columns.update(["input_sequence", "choice_string"])

        assert set(result_ds.column_names) == expected_columns
        assert len(result_ds) == 5
        result_ds = result_ds.sort('idx')
        correct_to_int = {k: i for i, k in enumerate(["A", "B", "C", "D", "E"])}

        def add_idx(ex, _idx):
            ex['idx'] = _idx
            ex['label'] = correct_to_int[ex['correct']]
            ex['choice_string'] = '\n'.join(map(
                lambda s: s.replace(')', ') '),
                ex['options']
            ))
            return ex

        ds = ds.map(
            add_idx,
            with_indices=True,
        ).sort('idx')

        for idx, (result, expected) in enumerate(zip(result_ds, ds)):
            assert result['idx'] == idx

            assert result['label'] == expected['label']

            if mode == TaskMode.QA:
                assert result['question'] == expected["question"]
                assert result['context'] == "Choices are: {}".format(expected['choice_string'])
            elif mode == TaskMode.ENTAILMENT:
                assert result['premise'] == expected['question']
                assert result['hypothesis'] == "Choices are: {}".format(expected['choice_string'])
            else:
                assert result['input_sequence'] == expected["question"]
                assert result['choice_string'] == expected['choice_string']
