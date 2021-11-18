from typing import Dict
import math

from src.preprocessors.task_preprocessor import TaskPreprocessor


@TaskPreprocessor.register('craigslist')
class CraigslistBargainsPreprocessor(TaskPreprocessor):

    def __init__(self, add_speaker_prefix: bool = False, fair_trade_tol=1e-3):
        super().__init__(
            choices=["seller", "buyer", "neither", "no deal", "unknown"],
            choice_str='"seller", "buyer", "neither","no deal", or "unknown"',
            mcq_choice_str='a) the seller\nb) the buyer\nc) neither - it is a'
                           ' fair compromise\nd) no deal\ne) unknown'
        )
        self.add_speaker_prefix = add_speaker_prefix
        self.fair_trade_tol = fair_trade_tol
        self.label_to_int = {
            "SELLER" : 0,
            "BUYER"  : 1,
            "NEITHER": 2,
            "NO-DEAL": 3,
            "UNKNOWN": 4
        }

    def __call__(self, data_instance, idx) -> Dict:

        output_dict = {
            "choices": self.choices,
            "idx"    : idx
        }

        # Get the utterances
        input_sequence = []
        for i, utterance in enumerate(data_instance['utterance']):
            prefix = ""
            if self.add_speaker_prefix:
                # Buyer always speaks first then seller, so on the even values
                # of i, the buyer will be speaking and on the odd values the
                # seller will be speaking.
                prefix = "Buyer" if i % 2 == 0 else "Seller"
                prefix += ": "

            input_sequence.append(f"{prefix}{utterance}")

        output_dict['input_sequence'] = "\n\n".join(input_sequence)
        label = self._get_label(
            data_instance['dialogue_acts'],
            data_instance['agent_info']['Target']
        )
        output_dict['label'] = self.label_to_int[label]
        output_dict['additional_inputs'] = [data_instance['items']["Price"][0]]

        return output_dict

    def _get_label(self, dialogue_acts, targets):
        intents, prices = dialogue_acts
        if not intents:
            return "UNKOWN"

        if intents[-1] != 'accept':
            return "NO-DEAL"

        final_price = -1
        for p in prices[::-1]:
            if p > -1:
                final_price = p
                break

        # Use less than 0 as it is impossible to have less than zero price.
        if final_price < 0:
            return "UNKNOWN"
        # Get both parties target price.
        buyer_target, seller_target = targets

        # Get how far off the final price was from both parties targets
        buyer_diff = abs(final_price - buyer_target)
        seller_diff = abs(final_price - seller_target)

        # Dealing with floats, so use isclose for safety with tol
        if math.isclose(seller_diff, buyer_diff, rel_tol=self.fair_trade_tol):
            return "NEITHER"
        elif seller_diff > buyer_diff:
            return "BUYER"
        elif seller_diff < buyer_diff:
            return "SELLER"

        raise ValueError("SOMETHING WENT VERY WRONG!")
