# [MODULE DOC]
# Dataset loader for the ViRL39K multimodal math RL dataset (parquet + local image
# folder). Builds a HuggingFace Dataset of chat-templated prompts, PIL images and boxed
# ground-truth answers for vision-language RL training.
# SPDX-License-Identifier: Apache-2.0

import math
import os

from datasets import load_dataset
from mathruler.grader import extract_boxed_content
from PIL import Image
from PIL.Image import Image as ImageObject


def convert_image(
    image: ImageObject,
    min_size: int | None = 28,
    max_pixels: int | None = None,
) -> ImageObject:
    # [FUNCTION DOC]
    # Upscales an image so its longer side is at least min_size (LANCZOS, preserving
    # aspect ratio), downscales it if its pixel count exceeds max_pixels (scale by sqrt
    # of ratio, default resampling), and converts to RGB mode. Inputs: image (PIL
    # Image), min_size (int, default 28), max_pixels (int or None = no cap). Returns the
    # processed PIL Image (possibly a new object). No side effects.
    # KNOWN ISSUE: min_size is typed as optional but passing None raises TypeError in
    # the comparison; the docstring says it accepts any 'image object' but
    # .size/.resize/.mode of a PIL image are required.
    """Convert and resize an image.

    Args:
        image: PIL Image or image object
        min_size: Minimum image size
        max_pixels: Maximum number of pixels (resize if exceeded)

    Returns:
        Processed PIL Image
    """

    width, height = image.size
    if max(width, height) < min_size:
        scale = max(min_size / width, min_size / height)
        new_width = int(round(width * scale))
        new_height = int(round(height * scale))
        image = image.resize((new_width, new_height), Image.LANCZOS)

    if max_pixels is not None and (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = (
            int(image.width * resize_factor),
            int(image.height * resize_factor),
        )
        image = image.resize((width, height))

    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def get_virl39k_rl_dataset(
    path: str,
    split: str,
    processor,
    max_length: int | None = None,
    img_folder_path: str | None = None,
):
    # [FUNCTION DOC]
    # Loads the parquet file at path (given split) with datasets.load_dataset, resolves
    # the image folder (defaults to the parquet's directory; raises ValueError if it
    # lacks an 'images' subfolder, only checked when defaulted), and maps each row
    # through process. Then drops the raw columns (question, pass rates, category,
    # source, qid, image) and, if max_length is set, filters out samples whose
    # processor-tokenized length (text + image tokens) exceeds max_length. Inputs: path,
    # split, processor (HF multimodal processor with tokenizer and image_processor),
    # max_length, img_folder_path. Returns the processed Dataset with columns messages,
    # messages_chat, images, answer (plus any remaining columns such as the original
    # 'answer' overwritten by the extracted one). Side effects: disk reads of parquet
    # and images.
    instruction_following = (
        r"You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
        r"The reasoning process MUST BE enclosed within <think> </think> tags. "
        r"The final answer MUST BE put in \boxed{}."
    )

    dataset = load_dataset("parquet", data_files=path, split=split)

    if img_folder_path is None:
        img_folder_path = os.path.dirname(path)
        if not os.path.isdir(os.path.join(img_folder_path, "images")):
            raise ValueError(f"images folder not found at {img_folder_path}")

    def process(example):
        # [FUNCTION DOC]
        # HELPER: Per-example mapper. Ensures the question contains an '<image>'
        # placeholder (prepends one if absent), appends the think/boxed instruction,
        # extracts the boxed answer from the answer field, opens and resizes every
        # referenced image (min 28, max 320x320 pixels), picks the model-specific image
        # token (Qwen vision tokens, Gemma3 boi_token, else processor.image_token), and
        # returns a dict with the chat-templated prompt string (messages, with
        # generation prompt, untokenized), an OpenAI-style messages_chat list (empty
        # image_url + text), the processed images and the answer.
        # KNOWN ISSUE: the 'processor is not None' guard in the fallback branch is
        # useless since processor.image_processor was already dereferenced;
        # messages_chat always contains exactly one image_url entry regardless of image
        # count.
        problem = example["question"]
        if "<image>" not in problem:
            problem = "<image>\n" + problem
        prompt = problem + " " + instruction_following
        answer = extract_boxed_content(example["answer"])

        img_path = [os.path.join(img_folder_path, img) for img in example["image"]]

        processed_images = [
            convert_image(Image.open(path_), min_size=28, max_pixels=320 * 320)
            for path_ in img_path
        ]

        image_processor_type = processor.image_processor.image_processor_type.lower()
        if "qwen" in image_processor_type:
            image_token = "<|vision_start|><|image_pad|><|vision_end|>"
        elif "gemma3" in image_processor_type:
            image_token = processor.boi_token
        else:
            image_token = processor.image_token if processor is not None else "<image>"

        messages = [{"role": "user", "content": prompt.replace("<image>", image_token)}]

        messages_chat = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": ""}},
                    {
                        "type": "text",
                        "text": prompt.replace("<image>", ""),
                    },
                ],
            }
        ]

        # Apply chat template
        messages = processor.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        return {
            "messages": messages,
            "messages_chat": messages_chat,
            "images": processed_images,
            "answer": answer,
        }

    dataset = dataset.map(process).remove_columns(
        [
            "question",
            "PassRate_32BTrained",
            "PassRate_7BBase",
            "category",
            "source",
            "qid",
            "image",
        ]
    )

    # Filter out sequences longer than max_length if max_length is provided
    if max_length is not None:

        def filter_length(sample):
            # [FUNCTION DOC]
            # HELPER: Filter predicate; runs the processor on the sample's templated
            # text and images (no padding, PyTorch tensors) and returns True if the
            # resulting input_ids length is <= max_length. CPU-heavy since it re-
            # processes images.
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
