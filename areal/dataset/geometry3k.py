# [MODULE DOC]
# Geometry3K vision-language math dataset loaders for SFT and RL. Images are center-
# cropped to a fixed size and JPEG-encoded; prompts have "<image>" replaced by the
# model-specific image token (Qwen vision tokens, Gemma3 boi_token, or
# processor.image_token).
# SPDX-License-Identifier: Apache-2.0

from io import BytesIO
from typing import Any

from datasets import load_dataset
from PIL import Image
from PIL.Image import Image as ImageObject
from torchvision import transforms


def pad_to_square(img: Image.Image, fill=(0, 0, 0)) -> Image.Image:
    # [FUNCTION DOC]
    # HELPER: Returns a new square PIL image of side max(width, height) filled with fill
    # (default black) with the input pasted centered. Not used elsewhere in this module.
    w, h = img.size
    side = max(w, h)
    new_img = Image.new(img.mode, (side, side), color=fill)
    offset = ((side - w) // 2, (side - h) // 2)
    new_img.paste(img, offset)
    return new_img


def convert_image(
    image: dict[str, Any] | ImageObject | str,
    fixed_width: int | None = None,
    fixed_height: int | None = None,
) -> ImageObject:
    # [FUNCTION DOC]
    # HELPER: If both fixed_width and fixed_height are given and the image size differs,
    # center-crops it to that size (torchvision CenterCrop). Converts to RGB if needed,
    # then encodes as JPEG and returns the raw bytes.
    # KNOWN ISSUE: Annotated to accept dict/str and return an Image, but it only works
    # with PIL images and returns JPEG bytes. CenterCrop expects (height, width) yet is
    # given (width, height); harmless only for square sizes.
    if (
        fixed_width is not None
        and fixed_height is not None
        and (image.width != fixed_width or image.height != fixed_height)
    ):
        preprocess = transforms.Compose(
            [
                transforms.CenterCrop((fixed_width, fixed_height)),  # <─ 核心操作
            ]
        )
        image = preprocess(image)
    if image.mode != "RGB":
        image = image.convert("RGB")
    with BytesIO() as output:
        image.save(output, format="JPEG")
        return output.getvalue()


def get_geometry3k_sft_dataset(
    path: str,
    split: str,
    processor,
    max_length: int | None = None,
):
    # [FUNCTION DOC]
    # Loads the split and, per example, swaps "<image>" for the image token, removes the
    # word "different" from the problem, converts images to 512x512 JPEG bytes and
    # builds seq = problem + answer + eos. Then runs the multimodal processor on seq and
    # images to get input_ids, mm_token_type_ids, multi_modal_input (pixel_values and
    # optional image_grid_thw), and a loss_mask of 1 over the last len(tokenized answer)
    # tokens and 0 elsewhere. Drops raw columns, optionally filters by max_length,
    # returns the Dataset.
    # KNOWN ISSUE: The loss mask counts answer tokens without eos, so the eos token is
    # masked as 1 and the first answer token may be masked 0; the "processor is not
    # None" check occurs after processor was already dereferenced.
    """
    "geometry3k": {
        "image_key": "images",
        "question_key": "problem",
        "answer_key": "answer"
    },
    """
    dataset = load_dataset(path=path, split=split)

    tokenizer = processor.tokenizer

    def process_example(example, idx):
        # [FUNCTION DOC]
        # HELPER: Per-example text/image preprocessing described above; idx is unused.
        # Add query_id column
        images = example["images"]
        image_processor_type = processor.image_processor.image_processor_type.lower()
        if "qwen" in image_processor_type:
            image_token = "<|vision_start|><|image_pad|><|vision_end|>"
        elif "gemma3" in image_processor_type:
            image_token = processor.boi_token
        else:
            image_token = processor.image_token if processor is not None else "<image>"
        example["problem"] = (
            example["problem"].replace("<image>", image_token).replace("different", "")
        )
        processed_images = []
        for image in images:
            processed_images.append(convert_image(image, 512, 512))
        example["images"] = processed_images
        example["seq"] = example["problem"] + example["answer"] + tokenizer.eos_token

        return example

    dataset = dataset.map(
        lambda example, idx: process_example(example, idx),
        with_indices=True,
    )

    def _process(example):
        # [FUNCTION DOC]
        # HELPER: Runs the processor on one example and adds input_ids,
        # mm_token_type_ids, multi_modal_input and loss_mask.
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
            multi_modal_input["image_grid_thw"] = processed_input[
                "image_grid_thw"
            ].squeeze(0)
        example["multi_modal_input"] = [multi_modal_input]
        answer_token = tokenizer.encode(example["answer"])
        loss_mask = [0] * (len(example["input_ids"]) - len(answer_token)) + [1] * len(
            answer_token
        )
        example["loss_mask"] = loss_mask
        return example

    dataset = dataset.map(
        lambda x: _process(x), remove_columns=["images", "seq", "problem", "answer"]
    )

    if max_length is not None:
        # Filter out sequences longer than max_length
        dataset = dataset.filter(lambda x: len(x["input_ids"]) <= max_length)

    return dataset


def get_geometry3k_rl_dataset(
    path: str,
    split: str,
    processor,
    max_length: int | None = None,
):
    # [FUNCTION DOC]
    # Loads the split and maps each sample to: messages (single user turn with image
    # token substituted, "different" removed and a think/boxed instruction appended,
    # rendered to a string via the tokenizer chat template with a generation prompt),
    # messages_chat (OpenAI-style content list with an empty image_url placeholder
    # filled later by the engine plus the text), and images (448x448 JPEG bytes).
    # Removes problem. If max_length is set, filters by the processor's total token
    # count including image tokens. Returns the Dataset.
    dataset = load_dataset(path=path, split=split)

    def process(sample):
        # [FUNCTION DOC]
        # HELPER: Builds messages, messages_chat and images for one sample.
        processed_images = [
            convert_image(image, 448, 448) for image in sample["images"]
        ]
        image_processor_type = processor.image_processor.image_processor_type.lower()
        if "qwen" in image_processor_type:
            image_token = "<|vision_start|><|image_pad|><|vision_end|>"
        elif "gemma3" in image_processor_type:
            image_token = processor.boi_token
        else:
            image_token = processor.image_token if processor is not None else "<image>"
        instruction_following = (
            r" You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
            r"The reasoning process MUST BE enclosed within <think> </think> tags. "
            r"The final answer MUST BE put in \boxed{}."
        )

        messages = [
            {
                "role": "user",
                "content": sample["problem"]
                .replace("<image>", image_token)
                .replace("different", "")
                + instruction_following,
            }
        ]
        messages_chat = [
            {
                "role": "user",
                "content": [
                    # NOTE: The prompt formatting with the image token `<image>` is not needed
                    # since the prompt will be processed automatically by the API server.
                    # Leave the "url" field empty. It will be filled in the engine.
                    {"type": "image_url", "image_url": {"url": ""}},
                    {
                        "type": "text",
                        "text": sample["problem"]
                        .replace("<image>", "")
                        .replace("different", "")
                        + instruction_following,
                    },
                ],
            }
        ]

        messages = processor.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        return {
            "messages": messages,
            "messages_chat": messages_chat,
            "images": processed_images,
        }

    dataset = dataset.map(process).remove_columns(["problem"])

    # Filter out sequences longer than max_length if max_length is provided
    if max_length is not None:

        def filter_length(sample):
            # [FUNCTION DOC]
            # HELPER: Runs the processor on the rendered prompt and images and returns
            # True if the input_ids length is <= max_length.
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
