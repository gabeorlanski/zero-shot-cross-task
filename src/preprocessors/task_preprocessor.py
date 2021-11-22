from typing import List, Dict
from copy import deepcopy
from src.common.registrable import Registrable
from enum import Enum, auto
import string


# Different tasks require different column. So in order to have the preprocessor
# convert the given task into a suitable format for a new task, we need to
# define "Modes" that control how the inputs are handled.
class TaskMode(Enum):
    CLASSIFICATION = auto()
    ENTAILMENT = auto()
    QA = auto()


class FixedChoiceTaskPreprocessor(Registrable):
    def __init__(
            self,
            choices: List = None,
            choice_str: str = None,
            mcq_choice_str: str = None,
            is_mcq: bool = False
    ):
        self.choices = choices

        if choice_str is None:
            choice_str = ', '.join(f'"{c}"' for c in choices[:-1])
            choice_str += f' or "{choices[-1]}"'

        if not mcq_choice_str:
            mcq_choice_str = '\n'.join(f'{letter}) {choice}'
                                       for letter, choice in zip(string.ascii_lowercase, choices))
            mcq_choice_str += '\n'

        self.choice_string = choice_str
        self.mcq_choice_string = mcq_choice_str
        self.is_mcq = is_mcq
        self.mode_specific_processors = {
            TaskMode.CLASSIFICATION: self.convert_to_classification,
            TaskMode.ENTAILMENT    : self.convert_to_entailment,
            TaskMode.QA            : self.convert_to_qa
        }

        self.mode = TaskMode.CLASSIFICATION
        self.mode_key_converter = self.convert_to_classification
        self._required_keys = {
            TaskMode.CLASSIFICATION: {
                "input_sequence": str,
            },
            TaskMode.ENTAILMENT    : {
                "premise"   : str,
                "hypothesis": str
            },
            TaskMode.QA            : {
                "question": str,
                "context" : str
            }
        }

    @property
    def required_keys_for_mode(self) -> Dict:
        """
        Required keys for the current mode.

        Returns: `Dict[str,type]`

        """
        return self._required_keys[self.mode]

    def set_mode(self, mode: TaskMode):
        """
        Set the mode of the processor. Will also set self.mode_key_converter to
         the corresponding function.

        Args:
            mode (TaskMode): The new mode to use.

        Returns: None
        """
        self.mode = mode
        self.mode_key_converter = self.mode_specific_processors[self.mode]

    def __call__(self, data_instance, idx) -> Dict:
        """
        Process a data instance

        Args:
            data_instance (Dict): instance to process
            idx (int): the idx of the instance.

        Returns (Dict): The processed instance.
        """
        processed_instance = self._process_instance(data_instance=data_instance, idx=idx)

        output = self.mode_key_converter(processed_instance)

        if not self.is_mcq:
            output['choice_string'] = self.choice_string
        else:
            output['choice_string'] = self.mcq_choice_string

        # Sanity check / validation
        self._validate_key(output, 'label', int)
        self._validate_key(output, 'domain', str)
        self._validate_key(output, 'idx', int)
        for key, key_type in self.required_keys_for_mode.items():
            self._validate_key(output, key, key_type)

        return output

    def _validate_key(self, d, key, key_type):
        if key not in d:
            raise KeyError(f"Missing required key '{key}' for mode {str(self.mode)}")
        elif not isinstance(d[key], key_type):
            raise TypeError(f"Key '{key}' has type {type(d['key'])}. Expected {key_type}")

    def _process_instance(self, data_instance, idx) -> Dict:
        raise NotImplementedError()

    def convert_to_classification(self, processed_instance: Dict) -> Dict:
        raise NotImplementedError()

    def convert_to_entailment(self, processed_instance: Dict) -> Dict:
        raise NotImplementedError()

    def convert_to_qa(self, processed_instance: Dict) -> Dict:
        raise NotImplementedError()
