from copy import deepcopy
from typing import Dict
import math

from src.preprocessors.task_preprocessor import FixedChoiceTaskPreprocessor


@FixedChoiceTaskPreprocessor.register('craigslist')
class CraigslistBargainsPreprocessor(FixedChoiceTaskPreprocessor):

    def __init__(
            self,
            add_speaker_prefix: bool = False,
            fair_trade_tol=1e-3,
            choices=None,
            question_str: str = "Who won the negotiation?",
            background_info_str: str = "The buyer wanted ${:.2f}, the seller wanted "
                                       "${:.2f}, and the final price is ${:.2f}",
            use_constant_domain: bool = False,
            is_mcq: bool = False,
            choice_str: str = None,
            mcq_choice_str:str=None
    ):
        if choices is None:
            choices = ["Seller", "Buyer", "Neither", "Unknown"]
        assert len(choices) == 4

        super().__init__(
            choices=choices,
            choice_str=choice_str,
            mcq_choice_str=mcq_choice_str,
            is_mcq=is_mcq
        )
        self.add_speaker_prefix = add_speaker_prefix
        self.fair_trade_tol = fair_trade_tol
        self.label_to_int = {
            "SELLER" : 0,
            "BUYER"  : 1,
            "NEITHER": 2,
            "UNKNOWN": 3
        }
        self.question_str = question_str
        self.background_info_str = background_info_str
        self.use_constant_domain = use_constant_domain

    def _process_instance(self, data_instance, idx) -> Dict:

        item_category = data_instance['items']['Category'][0]
        listing_price = data_instance['items']['Price'][0]
        output_dict = {
            "choices"      : self.choices,
            "idx"          : idx,
            "domain"       : "negotiations" if self.use_constant_domain else item_category,
            "listing_price": listing_price
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
        label, final_price, buyer_target, seller_target = self._get_label(
            data_instance['dialogue_acts']['intent'],
            data_instance['dialogue_acts']['price'],
            data_instance['agent_info']['Target']
        )
        output_dict['final_price'] = final_price
        output_dict['buyer_target'] = buyer_target
        output_dict['seller_target'] = seller_target
        output_dict['label'] = self.label_to_int[label]

        output_dict['additional_input_1'] = output_dict['listing_price']
        return output_dict

    def _get_label(self, intents, prices, targets):
        if not intents:
            return "UNKNOWN", -1, -1, -1

        final_price = -1
        # Get both parties target price.
        buyer_target, seller_target = targets
        found_intent = False
        for i, p in zip(intents[::-1], prices[::-1]):

            if i:
                found_intent = True
            if p > -1:
                final_price = p
                break

        # Use less than 0 as it is impossible to have less than zero price.
        if final_price < 0 or not found_intent:
            return "UNKNOWN", -1, buyer_target, seller_target

        if intents[-1] != 'accept':
            return "NEITHER", -1, buyer_target, seller_target

        # Get how far off the final price was from both parties targets
        buyer_diff = abs(final_price - buyer_target)
        seller_diff = abs(final_price - seller_target)

        # Dealing with floats, so use isclose for safety with tol
        if math.isclose(seller_diff, buyer_diff, rel_tol=self.fair_trade_tol):
            label = "NEITHER"
        elif seller_diff > buyer_diff:
            label = "BUYER"
        elif seller_diff < buyer_diff:
            label = "SELLER"
        else:
            raise ValueError("SOMETHING WENT VERY WRONG!")
        return label, final_price, buyer_target, seller_target

    def convert_to_classification(self, processed_instance: Dict) -> Dict:
        price_target_str = self.background_info_str.format(
            processed_instance['buyer_target'],
            processed_instance['seller_target'],
            processed_instance['final_price']
        )

        processed_instance['input_sequence'] = (
                price_target_str
                + ". "
                + processed_instance['input_sequence']
        )
        return processed_instance

    def convert_to_entailment(self, processed_instance: Dict) -> Dict:

        out = deepcopy(processed_instance)
        out['hypothesis'] = self.background_info_str.format(
            processed_instance['buyer_target'],
            processed_instance['seller_target'],
            processed_instance['final_price']
        )
        input_sequence = out.pop('input_sequence')
        out['premise'] = input_sequence

        return out

    def convert_to_qa(self, processed_instance: Dict) -> Dict:

        out = deepcopy(processed_instance)
        out['question'] = self.background_info_str.format(
            processed_instance['buyer_target'],
            processed_instance['seller_target'],
            processed_instance['final_price']
        )
        input_sequence = out.pop('input_sequence')
        out['context'] = input_sequence

        return out
