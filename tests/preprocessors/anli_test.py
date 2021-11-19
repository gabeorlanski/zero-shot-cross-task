import pytest
from datasets import load_dataset, set_caching_enabled

from src.preprocessors import anli, TaskMode


class TestAnliPreprocessor:

    def setup(self):
        set_caching_enabled(False)

    def teardown(self):
        set_caching_enabled(True)

    @pytest.mark.parametrize("mode", list(TaskMode), ids=list(map(str, TaskMode)))
    def test_call(self, mode):
        processor = anli.ANLIPreprocessor()
        processor.set_mode(mode)

        ds = load_dataset("anli", split='train_r1[:5]')

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
        if mode == TaskMode.CLASSIFICATION:
            expected_columns.add("input_sequence")
        elif mode == TaskMode.QA:
            expected_columns.update(["question", "context"])
        elif mode == TaskMode.ENTAILMENT:
            expected_columns.update(['premise', 'hypothesis'])

        assert set(result_ds.column_names) == expected_columns
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

            assert result['label'] == expected['label']

            expected_input_seq = f"Premise is '{expected['premise']}'. " \
                                 f"Hypothesis is '{expected['hypothesis']}'."

            if mode == TaskMode.QA:
                assert result['question'] == "Does the premise imply the hypothesis?"
                assert result['context'] == expected_input_seq
            elif mode == TaskMode.ENTAILMENT:
                assert result['premise'] == expected['premise']
                assert result['hypothesis'] == expected['hypothesis']
            else:
                assert result['input_sequence'] == expected_input_seq
