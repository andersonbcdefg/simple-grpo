"""
Module for loading LLMs and their tokenizers from huggingface.

"""

import torch
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from transformers.models.auto.processing_auto import AutoProcessor


def get_llm_tokenizer(
    model_name: str, device: str
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """
    Load and configure a language model and its tokenizer.

    Args:
        model_name: Name or path of the pretrained model to load
        device: Device to load the model on ('cpu' or 'cuda')

    Returns:
        tuple containing:
            - The loaded language model
            - The configured tokenizer for that model
    """

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )

    # default processer
    processor = AutoProcessor.from_pretrained(model_name)

    processor.tokenizer.pad_token = processor.tokenizer.eos_token
    model.config.pad_token_id = processor.tokenizer.pad_token_id

    processor.tokenizer.padding_side = "left"
    processor.padding_side = "left"

    # This fixed ~'need to set the pdadding.left' but even if you do that nothing works
    model.config.use_cache = False

    return model, processor
