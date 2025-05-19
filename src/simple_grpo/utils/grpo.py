from qwen_vl_utils import process_vision_info
import torch
from transformers.modeling_utils import PreTrainedModel
from simple_grpo.prompt import create_prompt


@torch.compile(dynamic=True)
def selective_log_softmax(logits, index):
    logprobs = logits.log_softmax(dim=-1)
    return torch.gather(logprobs, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)


####################################################################################
## Copied Directly from TRL -> generate log probs per token                 ########
## https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py ########
####################################################################################
def get_per_token_logps(
    model: PreTrainedModel, input_ids, attention_mask, logits_to_keep
):
    # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
    logits = model.__call__(
        input_ids=input_ids,
        attention_mask=attention_mask,
        logits_to_keep=logits_to_keep + 1,
    ).logits
    logits = logits[
        :, :-1, :
    ]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred

    input_ids = input_ids[:, -logits_to_keep:]
    # For transformers<=4.48, logits_to_keep argument isn't supported, so here we drop logits ourselves.
    # See https://github.com/huggingface/trl/issues/2770
    logits = logits[:, -logits_to_keep:]
    return selective_log_softmax(
        logits, input_ids
    )  #  compute logprobs for the input tokens


def get_per_token_logps_vl(
    model, input_ids, attention_mask, image_path, tokenizer, logits_to_keep, prompt
):
    """
    We have the input ids - all the correct tokens including all chate templates/special tokens etc
    We just need to include the image - and have the same sort of obj to pass to the model to generate
    the logits

    So lets generate with the image


    resulting to a very non-generic way to do this - TODO: make this better
    """
    conversation = create_prompt(image_path, prompt)

    text = tokenizer.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False, padding_side="left"
    )
    image_inputs, video_inputs = process_vision_info(conversation)  # type: ignore -

    prompt_inputs = (
        tokenizer(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            padding_side="left",
        )
        .to(model.device)
        .to(model.dtype)
    )

    # Repeat input tensors for batch generation
    batched_prompt_inputs = {}
    for key, value in prompt_inputs.items():
        if torch.is_tensor(value):
            batched_prompt_inputs[key] = value.repeat(
                input_ids.shape[0], *([1] * (value.dim() - 1))
            )
        else:
            # Handle non-tensor items if necessary, otherwise just copy
            batched_prompt_inputs[key] = value

    batched_prompt_inputs["input_ids"] = input_ids
    batched_prompt_inputs["attention_mask"] = attention_mask

    # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
    # logits = model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=logits_to_keep + 1).logits
    logits = model(**batched_prompt_inputs).logits
    logits = logits[
        :, :-1, :
    ]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred

    input_ids = input_ids[:, -logits_to_keep:]
    logits = logits[:, -logits_to_keep:]
    return selective_log_softmax(
        logits, input_ids
    )  #  compute logprobs for the input tokens
