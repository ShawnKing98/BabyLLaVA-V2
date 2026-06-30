#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, \
                         GPT2Config, GPT2Model, GPT2LMHeadModel, PreTrainedTokenizerFast

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM


class LlavaGPT2Config(GPT2Config):
    model_type = "llava_gpt2"


# class LlavaGPT2Tokenizer(PreTrainedTokenizerFast):
#     def __call__(
#         self,
#         text = None,
#         add_bos_eos: bool = True,
#         *args, **kwargs):
        
#         def add_bos_eos_tokens(input_text):
#             if isinstance(input_text, str):
#                 return f"<sos> {input_text} <eos>"
#             elif isinstance(input_text, (list, tuple)):
#                 return [f"<sos> {t} <eos>" if isinstance(t, str) else t for t in input_text]
#             return input_text
        
#         if add_bos_eos and text is not None:
#             text = add_bos_eos_tokens(text)

#         return super().__call__(text, *args, **kwargs)


class LlavaGPT2Model(LlavaMetaModel, GPT2Model):
    config_class = LlavaGPT2Config

    def __init__(self, config: GPT2Config):
        super(LlavaGPT2Model, self).__init__(config)
    
    @property
    def embed_tokens(self):
        return self.get_input_embeddings() 


class LlavaGPT2ForCausalLM(GPT2LMHeadModel, LlavaMetaForCausalLM):
    config_class = LlavaGPT2Config

    def __init__(self, config):
        super(GPT2LMHeadModel, self).__init__(config)
        self.transformer = LlavaGPT2Model(config)
        # self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.model_parallel = False
        self.positional_embedding_type = getattr(config, 'positional_embedding_type', 'learned_absolute')
        if self.positional_embedding_type != 'learned_absolute' and getattr(self.transformer, "wpe", None) is not None:
            print(f"Setting positional embedding type to {self.positional_embedding_type}, taking wpe from the language backbone and freezing it.")
            for p in self.transformer.wpe.parameters():
                p.requires_grad = False
        else:
            print(f"Keep using learned absolute positional embeddings.")
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    @property
    def model(self):
        return self.transformer

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
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
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
                images,
                image_sizes
            )
        
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            if inputs.shape[-1] == 1:    # inputs is a single <image> token
                inputs_embeds = self.encode_images(images)
                if attention_mask is not None:
                    attention_mask = torch.ones(inputs_embeds.shape[0:2], dtype=torch.bool, device=inputs_embeds.device)
            else:
                (
                    inputs,
                    position_ids,
                    attention_mask,
                    _,
                    inputs_embeds,
                    _
                ) = self.prepare_inputs_labels_for_multimodal(
                    inputs,
                    position_ids,
                    attention_mask,
                    None,
                    None,
                    images,
                    image_sizes=image_sizes
                )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs

AutoConfig.register("llava_gpt2", LlavaGPT2Config)
AutoModelForCausalLM.register(LlavaGPT2Config, LlavaGPT2ForCausalLM)
# AutoTokenizer.register(LlavaGPT2Config, None, LlavaGPT2Tokenizer)
