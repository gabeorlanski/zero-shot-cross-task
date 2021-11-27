from copy import deepcopy
from typing import Dict
import math

from src.preprocessors.task_preprocessor import FixedChoiceTaskPreprocessor


@FixedChoiceTaskPreprocessor.register('aqua_rat')
class AquaRatPreprocessor(FixedChoiceTaskPreprocessor):

    def __init__(
            self,
            use_lowercase_choices: bool = False,
            is_mcq: bool = False,
            choice_str: str = None,
            mcq_choice_str: str = None
    ):
        self.use_lowercase_choices = use_lowercase_choices
        if self.use_lowercase_choices:
            choices = ["a", "b", "c", "d", "e"]
        else:
            choices = ["A", "B", "C", "D", "E"]

        classification_template = "{}"
        premise_template = "{}"
        hypothesis_template = "Choices are: {}"
        context_template = "Choices are: {}"
        question_template = "{}"

        super().__init__(
            choices=choices,
            classification_template=classification_template,
            premise_template=premise_template,
            hypothesis_template=hypothesis_template,
            question_template=question_template,
            context_template=context_template,
            choice_str=choice_str,
            mcq_choice_str=mcq_choice_str,
            is_mcq=is_mcq
        )

        self._correct_to_int = {k: i for i, k in enumerate(["A", "B", "C", "D", "E"])}

    def _process_instance(self, data_instance, idx) -> Dict:
        output_dict = {
            "choices" : self.choices,
            "idx"     : idx,
            "domain"  : "math problem",
            "question": data_instance['question'],
            "label"   : self._correct_to_int[data_instance['correct']]
        }

        choice_strs = []

        for choice in data_instance['options']:

            if self.use_lowercase_choices:
                choice_letter, answer = choice.split(")")
                choice_str = f"{choice_letter.lower()}) {answer}"
            else:
                choice_str = choice.replace(")", ") ")
            choice_strs.append(choice_str)

        output_dict['choice_string'] = "\n".join(choice_strs)
        return output_dict

    def convert_to_classification(self, processed_instance: Dict) -> Dict:
        processed_instance['input_sequence'] = processed_instance.pop('question')
        return processed_instance

    def convert_to_entailment(self, processed_instance: Dict) -> Dict:

        processed_instance['hypothesis'] = self.hypothesis_template.format(
            processed_instance.pop('choice_string')
        )
        processed_instance['premise'] = self.premise_template.format(
            processed_instance.pop('question')
        )
        return processed_instance

    def convert_to_qa(self, processed_instance: Dict) -> Dict:
        processed_instance['question'] = processed_instance.pop('question')
        processed_instance['context'] = self.context_template.format(
            processed_instance.pop('choice_string'))
        return processed_instance
