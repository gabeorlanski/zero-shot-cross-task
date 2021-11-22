from copy import deepcopy
from typing import Dict
import math

from src.preprocessors.task_preprocessor import FixedChoiceTaskPreprocessor


@FixedChoiceTaskPreprocessor.register('anli')
class ANLIPreprocessor(FixedChoiceTaskPreprocessor):
    def __init__(
            self,
            choices=None,
            is_mcq: bool = False,
            choice_str: str = None,
            mcq_choice_str: str = None
    ):
        if choices is None:
            choices = ["Yes", "Maybe", "No"]
        assert len(choices) == 3

        super().__init__(
            choices=choices,
            choice_str=choice_str,
            mcq_choice_str=mcq_choice_str,
            is_mcq=is_mcq
        )

    def _process_instance(self, data_instance, idx) -> Dict:
        return {
            "choices"   : self.choices,
            "idx"       : idx,
            "domain"    : "entailment",
            "premise"   : data_instance['premise'],
            "label"     : data_instance['label'],
            "hypothesis": data_instance['hypothesis'],
        }

    def convert_to_classification(self, processed_instance: Dict) -> Dict:
        out = deepcopy(processed_instance)
        premise = out.pop('premise')
        hypothesis = out.pop('hypothesis')
        out['input_sequence'] = f"Premise is '{premise}'. " \
                                f"Hypothesis is '{hypothesis}'."
        return out

    def convert_to_entailment(self, processed_instance: Dict) -> Dict:
        return processed_instance

    def convert_to_qa(self, processed_instance: Dict) -> Dict:
        out = deepcopy(processed_instance)
        premise = out.pop('premise')
        hypothesis = out.pop('hypothesis')
        out['context'] = f"Premise is '{premise}'. " \
                         f"Hypothesis is '{hypothesis}'."
        out['question'] = "Does the premise imply the hypothesis?"
        return out
