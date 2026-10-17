# [MODULE DOC]
# Small PIL image helpers: base64 PNG encoding and centered padding to a common size.
# SPDX-License-Identifier: Apache-2.0

import base64
from io import BytesIO

from PIL import Image
from PIL.Image import Image as ImageObject


def image2base64(images: list[ImageObject] | ImageObject) -> list[str]:
    # [FUNCTION DOC]
    # Encodes one image or a list of PIL images as PNG and returns a list of base64
    # utf-8 strings (always a list, even for a single image).
    if isinstance(images, ImageObject):
        images = [images]

    byte_images = []
    for image in images:
        with BytesIO() as buffer:
            image.save(buffer, format="PNG")
            buffer.seek(0)
            byte_image = base64.b64encode(buffer.read()).decode("utf-8")
            byte_images.append(byte_image)

    return byte_images


def pad_images_batch_to_max_size(images):
    # [FUNCTION DOC]
    # Pads every image to the batch's max width and height by pasting it centered onto a
    # black RGB canvas. Returns new images. Raises ValueError on an empty list (max of
    # empty).
    max_width = max(image.size[0] for image in images)
    max_height = max(image.size[1] for image in images)

    padded_images = []

    for image in images:
        width, height = image.size

        padding_left = (max_width - width) // 2
        padding_top = (max_height - height) // 2

        padded_image = Image.new("RGB", (max_width, max_height), (0, 0, 0))
        padded_image.paste(image, (padding_left, padding_top))

        padded_images.append(padded_image)

    return padded_images
