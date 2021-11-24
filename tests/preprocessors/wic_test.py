import pytest
from datasets import load_dataset, set_caching_enabled

from src.preprocessors import wic, TaskMode


class TestEntailmentPreprocessor:

    def setup(self):
        set_caching_enabled(False)

    def teardown(self):
        set_caching_enabled(True)

    @pytest.mark.parametrize("mode", list(TaskMode), ids=list(map(str, TaskMode)))
    def test_call(self, mode):

        processor = wic.WICPreprocessor()
        ds = load_dataset("super_glue", "wic", split='train[:5]')
        processor.set_mode(mode)

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

            sentence_str = f"Sentence 1: {expected['sentence1']}. " \
                           f"Sentence 2: {expected['sentence2']}."

            if mode == TaskMode.QA:
                assert result['question'] == f'Does the word "{expected["word"]}" ' \
                                             f'have the same meaning in both?'
                assert result['context'] == sentence_str
            elif mode == TaskMode.ENTAILMENT:
                assert result['premise'] == sentence_str
                assert result['hypothesis'] == f'The word "{expected["word"]}" ' \
                                               f'has the same meaning in both.'
            else:
                assert result['input_sequence'] == (
                        sentence_str
                        + f' The word "{expected["word"]}" '
                          f'has the same meaning in both.'
                )
