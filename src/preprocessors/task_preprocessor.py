from typing import List, Dict

from overrides import overrides
from src.common.registrable import Registrable


class TaskPreprocessor(Registrable):
    def __init__(
            self,
            choices: List = None,
            choice_str: str = None,
            mcq_choice_str: str = None
    ):
        self.choices = choices
        self.choice_string = choice_str
        self.mcq_choice_string = mcq_choice_str

    def __call__(self, data_instance, idx) -> Dict:
        raise NotImplementedError()
