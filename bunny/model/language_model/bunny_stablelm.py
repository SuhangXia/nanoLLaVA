from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM

from bunny.model.language_model.stable_lm.modeling_stablelm_epoch import StableLMEpochModel, StableLMEpochConfig, \
    StableLMEpochForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast

from bunny.model.bunny_arch import BunnyMetaModel, BunnyMetaForCausalLM, ActionHead
from bunny.constants import IGNORE_INDEX


class BunnyStableLMConfig(StableLMEpochConfig):
    model_type = "bunny-stablelm"


class BunnyStableLMModel(BunnyMetaModel, StableLMEpochModel):
    config_class = BunnyStableLMConfig

    def __init__(self, config: StableLMEpochConfig):
        super(BunnyStableLMModel, self).__init__(config)


class BunnyStableLMForCausalLM(StableLMEpochForCausalLM, BunnyMetaForCausalLM):
    config_class = BunnyStableLMConfig

    def __init__(self, config):
        super(StableLMEpochForCausalLM, self).__init__(config)
        self.model = BunnyStableLMModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.action_head = ActionHead(config.hidden_size, action_dim=7)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            images: Optional[torch.FloatTensor] = None,
            return_dict: Optional[bool] = None,
            num_image_tokens: Optional[int] = None,
            action_labels: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images
            )

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        action_prediction = None
        if num_image_tokens is not None and num_image_tokens > 0:
            last_im_hidden = hidden_states[:, num_image_tokens - 1, :]
            action_prediction = self.action_head(last_im_hidden)

        loss = None
        if action_labels is not None and action_prediction is not None:
            loss = F.mse_loss(action_prediction, action_labels)
        elif labels is not None:
            shift_logits = logits[..., :-1, :].contiguous().view(-1, self.config.vocab_size)
            shift_labels = labels[..., 1:].contiguous().view(-1).to(shift_logits.device)
            loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=IGNORE_INDEX)

        if not return_dict:
            out = (logits,) + outputs[1:]
            out = ((loss,) + out) if loss is not None else out
            if action_prediction is not None:
                out = out + (action_prediction,)
            return out

        out = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        out.action_prediction = action_prediction
        return out

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, attention_mask=None,
                                      **kwargs):
        images = kwargs.pop("images", None)

        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, attention_mask=attention_mask,
            **kwargs
        )

        if images is not None:
            _inputs['images'] = images
        return _inputs


AutoConfig.register("bunny-stablelm", BunnyStableLMConfig)
AutoModelForCausalLM.register(BunnyStableLMConfig, BunnyStableLMForCausalLM)
