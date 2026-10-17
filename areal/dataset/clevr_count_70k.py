# [MODULE DOC]
# CLEVR-Count-70K vision counting dataset loaders for SFT and RL. Images are downscaled
# to at most 336*336 pixels and JPEG-encoded. When torch.distributed is initialized,
# rank 0 preprocesses first with multiprocessing (up to DATASET_NUM_PROC=16 workers) to
# populate the HF cache, a barrier follows, then every rank reruns preprocessing
# (hitting the cache). Uses the "CLEVR70KDataset" logger.
# SPDX-License-Identifier: Apache-2.0

import math
import os
from io import BytesIO
from typing import Any

import torch.distributed as dist
from datasets import load_dataset
from PIL.Image import Image as ImageObject

from areal.utils import logging

logger = logging.getLogger("CLEVR70KDataset")

DATASET_NUM_PROC = 16


def convert_image(
    image: dict[str, Any] | ImageObject | str,
    max_pixels: int | None,
) -> ImageObject:
    # [FUNCTION DOC]
    # HELPER: If max_pixels is given and the image has more pixels, resizes it by
    # sqrt(max_pixels / area) keeping aspect ratio. Converts to RGB if needed and
    # returns JPEG-encoded bytes.
    # KNOWN ISSUE: Annotated to accept dict/str and return an Image, but it requires a
    # PIL image and returns bytes.
    if max_pixels is not None and (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = (
            int(image.width * resize_factor),
            int(image.height * resize_factor),
        )
        image = image.resize((width, height))

    if image.mode != "RGB":
        image = image.convert("RGB")
    with BytesIO() as output:
        image.save(output, format="JPEG")
        return output.getvalue()


def get_clevr_count_70k_sft_dataset(
    path: str,
    split: str,
    processor,
    max_length: int | None = None,
):
    # [FUNCTION DOC]
    # Builds the SFT dataset via the nested _do_preprocess using the distributed
    # rank-0-first caching scheme described in the module doc (logs a warning to put
    # HF_HOME on shared storage). Returns a Dataset with input_ids, mm_token_type_ids,
    # multi_modal_input and loss_mask.
    """
    "clevr_count_70k": {
        "image_key": "images",
        "question_key": "problem",
        "answer_key": "answer"
    },
    """

    def _do_preprocess(
        path: str,
        split: str,
        processor,
        max_length: int | None = None,
        num_proc: int | None = None,
    ):
        # [FUNCTION DOC]
        # HELPER: Loads the split; maps each example to substitute the model image token
        # for "<image>", remove "different", convert images and build seq = problem +
        # answer + eos; then runs the processor to produce input_ids, mm_token_type_ids,
        # multi_modal_input (pixel_values, optional image_grid_thw) and a loss_mask of 1
        # over the trailing len(tokenized answer) tokens. Drops raw columns and
        # optionally filters by max_length. Returns the Dataset.
        # KNOWN ISSUE: The num_proc argument is ignored and overwritten with
        # min(cpu_count, 16), so multiprocessing is used even in the non-distributed
        # path that intends num_proc=None. Loss mask ignores the trailing eos token so
        # the mask is shifted by one.
        dataset = load_dataset(path=path, split=split)

        tokenizer = processor.tokenizer

        def process_example(example):
            # [FUNCTION DOC]
            # HELPER: Per-example text and image preprocessing building the seq field.
            # Add query_id column
            images = example["images"]
            image_processor_type = (
                processor.image_processor.image_processor_type.lower()
            )
            if "qwen" in image_processor_type:
                image_token = "<|vision_start|><|image_pad|><|vision_end|>"
            elif "gemma3" in image_processor_type:
                image_token = processor.boi_token
            else:
                image_token = (
                    processor.image_token if processor is not None else "<image>"
                )
            example["problem"] = (
                example["problem"]
                .replace("<image>", image_token)
                .replace("different", "")
            )
            processed_images = []
            for image in images:
                processed_images.append(convert_image(image, 336 * 336))
            example["images"] = processed_images
            example["seq"] = (
                example["problem"] + example["answer"] + tokenizer.eos_token
            )

            return example

        num_proc = max(1, min(os.cpu_count(), 16))
        dataset = dataset.map(
            lambda example: process_example(example),
            num_proc=num_proc,
        )

        def _process(example):
            # [FUNCTION DOC]
            # HELPER: Runs the multimodal processor on one example and adds token ids,
            # multimodal inputs and loss mask.
            text = example["seq"]
            processed_input = processor(
                text=[text],
                images=example["images"],
                padding=False,
                return_tensors="pt",
                return_length=True,
                return_attention_mask=False,
            )

            example["input_ids"] = processed_input["input_ids"].squeeze(0)
            example["mm_token_type_ids"] = processed_input.get("token_type_ids", None)
            multi_modal_input = {}
            multi_modal_input["pixel_values"] = processed_input["pixel_values"]
            if "image_grid_thw" in processed_input:
                multi_modal_input["image_grid_thw"] = processed_input["image_grid_thw"]
            example["multi_modal_input"] = [multi_modal_input]
            answer_token = tokenizer.encode(example["answer"])
            loss_mask = [0] * (len(example["input_ids"]) - len(answer_token)) + [
                1
            ] * len(answer_token)
            example["loss_mask"] = loss_mask
            return example

        dataset = dataset.map(
            lambda x: _process(x),
            remove_columns=["images", "seq", "problem", "answer"],
            num_proc=num_proc,
        )

        if max_length is not None:
            # Filter out sequences longer than max_length
            dataset = dataset.filter(lambda x: len(x["input_ids"]) <= max_length)

        return dataset

    if dist.is_initialized():
        # Use multi-processing to accelerate data-processing
        # FIXME: processor process data extremely slowly in transformers > 4.53.1
        num_proc = max(1, min(os.cpu_count(), DATASET_NUM_PROC))
        logger.warning("Please set HF_HOME to your NFS directory")
        if int(os.getenv("RANK", "0")) == 0:
            # First process data in rank 0, and use HF cache to load pre-processed dataset in other ranks
            dataset = _do_preprocess(path, split, processor, max_length, num_proc)
        dist.barrier()
    else:
        # Do not use multi-processing (slow)
        num_proc = None

    # If use multiprocessing, it will load dataset in HF cache
    dataset = _do_preprocess(path, split, processor, max_length, num_proc)
    return dataset


def get_clevr_count_70k_rl_dataset(
    path: str,
    split: str,
    processor,
    max_length: int | None = None,
):
    # [FUNCTION DOC]
    # Builds the RL dataset via the nested _do_preprocess with the same rank-0-first
    # caching scheme as the SFT loader. Returns a Dataset with messages (chat-template-
    # rendered prompt string) and images.
    def _do_preprocess(
        path: str,
        split: str,
        processor,
        max_length: int | None = None,
        num_proc: int | None = None,
    ):
        # [FUNCTION DOC]
        # HELPER: Loads the split and maps each sample (with num_proc) to a system
        # prompt asking to count items and answer as a number in [ ], plus a user turn
        # with the image token substituted and "different" removed, rendered with the
        # tokenizer chat template and a generation prompt; images are converted to JPEG
        # bytes. Removes problem and optionally filters by processor token count
        # including image tokens. Returns the Dataset.
        dataset = load_dataset(path=path, split=split)

        def process(sample):
            # [FUNCTION DOC]
            # HELPER: Builds the rendered messages string and processed images for one
            # sample.
            processed_images = [
                convert_image(image, 336 * 336) for image in sample["images"]
            ]
            image_processor_type = (
                processor.image_processor.image_processor_type.lower()
            )
            if "qwen" in image_processor_type:
                image_token = "<|vision_start|><|image_pad|><|vision_end|>"
            elif "gemma3" in image_processor_type:
                image_token = processor.boi_token
            else:
                image_token = (
                    processor.image_token if processor is not None else "<image>"
                )
            system_prompt = {
                "role": "system",
                "content": (
                    "Solve the following question: count the number of items in the image and provide the final answer in [ ] format, ensuring that only the number is inside the brackets without any additional text or explanations. "
                ),
            }

            messages = [
                {
                    "role": "user",
                    "content": sample["problem"]
                    .replace("<image>", image_token)
                    .replace("different", ""),
                }
            ]
            messages.insert(0, system_prompt)
            messages = processor.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            return {"messages": messages, "images": processed_images}

        dataset = dataset.map(process, num_proc=num_proc).remove_columns(["problem"])

        # Filter out sequences longer than max_length if max_length is provided
        if max_length is not None:

            def filter_length(sample):
                # [FUNCTION DOC]
                # HELPER: True if the processor's input_ids length for the prompt and
                # images is <= max_length.
                # Process the sample to get the total token count including image tokens
                processed_input = processor(
                    text=[sample["messages"]],
                    images=sample["images"],
                    padding=False,
                    return_tensors="pt",
                    return_length=True,
                    return_attention_mask=False,
                )
                total_tokens = len(processed_input["input_ids"].squeeze(0))
                return total_tokens <= max_length

            dataset = dataset.filter(filter_length)
        return dataset

    if dist.is_initialized():
        # Use multi-processing to accelerate data-processing
        # FIXME: processor process data extremely slowly in transformers > 4.53.1
        num_proc = max(1, min(os.cpu_count(), DATASET_NUM_PROC))
        logger.warning("Please set HF_HOME to your NFS directory")
        if int(os.getenv("RANK", "0")) == 0:
            # First process data in rank 0, and use HF cache to load pre-processed dataset in other ranks
            dataset = _do_preprocess(path, split, processor, max_length, num_proc)
        dist.barrier()
    else:
        # Do not use multi-processing (slow)
        num_proc = None

    # If use multiprocessing, it will load dataset in HF cache
    dataset = _do_preprocess(path, split, processor, max_length, num_proc)
    return dataset
