# [MODULE DOC]
# Round-trip tests for RPC serialize_value/deserialize_value covering primitives,
# collections, torch tensors (incl. bfloat16 and meta), numpy arrays, PIL images,
# dataclasses, HF tokenizers/processors (loaded from local storage or the HF hub),
# nested mixes, and CUDA-to-CPU conversion.
"""Tests for RPC serialization utilities."""

from dataclasses import dataclass
from io import BytesIO

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import AutoTokenizer

from tests.utils import get_model_path

from areal.infra.rpc.serialization import deserialize_value, serialize_value


@dataclass
class SampleData:
    # [CLASS DOC]
    # HELPER: Simple dataclass (name str, value int, tensor) used to test dataclass
    # serialization.
    name: str
    value: int
    tensor: torch.Tensor


class TestSerializationRoundTrip:
    # [CLASS DOC]
    # Groups the serialization round-trip tests.
    """Test serialization/deserialization for all supported types."""

    def test_primitives_and_collections(self):
        # [FUNCTION DOC]
        # Asserts a dict of primitives, lists and nested dicts round-trips to an equal
        # value.
        """Test primitives and basic collections."""
        payload = {
            "str": "hello",
            "int": 42,
            "float": 3.14,
            "bool": True,
            "none": None,
            "list": [1, 2, 3],
            "nested": {"a": 1, "b": [2, 3]},
        }
        serialized = serialize_value(payload)
        deserialized = deserialize_value(serialized)
        assert deserialized == payload

    def test_tensors(self):
        # [FUNCTION DOC]
        # For int64, float32, bfloat16 and meta tensors asserts serialized type "tensor"
        # and that the round-trip keeps dtype, shape and values (or meta-ness for meta
        # tensors).
        """Test torch tensors including special dtypes."""
        tensors = {
            "int": torch.tensor([1, 2, 3], dtype=torch.int64),
            "float32": torch.randn(3, 4, dtype=torch.float32),
            "bfloat16": torch.randn(2, 3, dtype=torch.bfloat16),
            "meta": torch.empty(5, 10, dtype=torch.float32, device="meta"),
        }

        for name, original in tensors.items():
            serialized = serialize_value(original)
            assert serialized["type"] == "tensor"
            deserialized = deserialize_value(serialized)

            if original.is_meta:
                assert deserialized.is_meta
            else:
                assert torch.equal(deserialized, original)
            assert deserialized.dtype == original.dtype
            assert deserialized.shape == original.shape

    def test_numpy_arrays(self):
        # [FUNCTION DOC]
        # For int32, float64 and bool arrays asserts type "ndarray" and strict equality
        # with dtype.
        """Test NumPy array serialization."""
        arrays = {
            "int": np.array([1, 2, 3], dtype=np.int32),
            "float": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
            "bool": np.array([True, False], dtype=np.bool_),
        }

        for name, original in arrays.items():
            serialized = serialize_value(original)
            assert serialized["type"] == "ndarray"
            deserialized = deserialize_value(serialized)
            np.testing.assert_array_equal(deserialized, original, strict=True)
            assert deserialized.dtype == original.dtype

    def test_numpy_object_array_rejected(self):
        # [FUNCTION DOC]
        # Asserts serializing an object-dtype array raises ValueError "Object or void
        # dtype".
        """Object arrays should be rejected."""
        array = np.array([{"a": 1}], dtype=object)
        with pytest.raises(ValueError, match="Object or void dtype"):
            serialize_value(array)

    def test_pil_image_roundtrip(self):
        # [FUNCTION DOC]
        # Asserts an RGB PIL image serializes as "pil_image" and round-trips with same
        # size, mode and identical PNG-encoded bytes.
        """Test PIL image serialization for VLM RPC payloads."""
        original = Image.new("RGB", (8, 6), color=(12, 34, 56))

        serialized = serialize_value(original)
        assert serialized["type"] == "pil_image"

        deserialized = deserialize_value(serialized)
        assert isinstance(deserialized, Image.Image)
        assert deserialized.size == original.size
        assert deserialized.mode == original.mode

        # Validate pixel content with a deterministic byte compare
        with BytesIO() as o_buf, BytesIO() as d_buf:
            original.save(o_buf, format="PNG")
            deserialized.save(d_buf, format="PNG")
            assert o_buf.getvalue() == d_buf.getvalue()

    def test_dataclass(self):
        # [FUNCTION DOC]
        # Asserts a SampleData instance serializes as "dataclass" with a class_path
        # ending in SampleData and round-trips with equal fields and tensor.
        """Test dataclass serialization with nested tensors."""
        original = SampleData(
            name="test",
            value=42,
            tensor=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        )

        serialized = serialize_value(original)
        assert serialized["type"] == "dataclass"
        assert serialized["class_path"].endswith("SampleData")

        deserialized = deserialize_value(serialized)
        assert isinstance(deserialized, SampleData)
        assert deserialized.name == original.name
        assert deserialized.value == original.value
        assert torch.equal(deserialized.tensor, original.tensor)

    def test_tokenizer(self):
        # [FUNCTION DOC]
        # Loads the Qwen3-0.6B tokenizer (network or local path), asserts type
        # "tokenizer" and that the round-tripped tokenizer has the same vocab size and
        # encodes "test" identically.
        """Test Hugging Face tokenizer serialization."""
        original = AutoTokenizer.from_pretrained(
            get_model_path(
                "/storage/openpsi/models/Qwen__Qwen3-0.6B", "Qwen/Qwen3-0.6B"
            )
        )

        serialized = serialize_value(original)
        assert serialized["type"] == "tokenizer"

        deserialized = deserialize_value(serialized)
        assert deserialized.vocab_size == original.vocab_size
        assert deserialized.encode("test") == original.encode("test")

    def test_processor(self):
        # [FUNCTION DOC]
        # Loads the Qwen2.5-VL-3B processor, asserts type "processor" and equal
        # tokenizer vocab size and encoding after round-trip.
        """Test Hugging Face processor serialization."""
        from transformers import AutoProcessor

        original = AutoProcessor.from_pretrained(
            get_model_path(
                "/storage/openpsi/models/Qwen__Qwen2.5-VL-3B-Instruct",
                "Qwen/Qwen2.5-VL-3B-Instruct",
            )
        )

        serialized = serialize_value(original)
        assert serialized["type"] == "processor"

        deserialized = deserialize_value(serialized)
        assert deserialized.tokenizer.vocab_size == original.tokenizer.vocab_size
        assert deserialized.tokenizer.encode("test") == original.tokenizer.encode(
            "test"
        )

    def test_nested_structure(self):
        # [FUNCTION DOC]
        # Round-trips a dict mixing tensor, ndarray, dataclass, list of tensor/array and
        # nested dict; asserts key values are preserved.
        """Test complex nested structure with multiple types."""
        payload = {
            "tensor": torch.tensor([1.0, 2.0, 3.0]),
            "array": np.array([4, 5, 6]),
            "dataclass": SampleData(
                name="nested",
                value=7,
                tensor=torch.zeros(2, 2),
            ),
            "list": [torch.ones(3), np.zeros(2)],
            "meta": {"text": "value"},
        }

        serialized = serialize_value(payload)
        deserialized = deserialize_value(serialized)

        assert torch.equal(deserialized["tensor"], payload["tensor"])
        np.testing.assert_array_equal(deserialized["array"], payload["array"])
        assert isinstance(deserialized["dataclass"], SampleData)
        assert deserialized["dataclass"].name == "nested"
        assert torch.equal(deserialized["list"][0], payload["list"][0])
        assert deserialized["meta"]["text"] == "value"

    @pytest.mark.skipif(
        not hasattr(torch, "cuda") or not torch.cuda.is_available(),
        reason="CUDA not available",
    )
    def test_cuda_tensors_moved_to_cpu(self):
        # [FUNCTION DOC]
        # Skipped without CUDA. Asserts a CUDA tensor deserializes on CPU with equal
        # values.
        """CUDA tensors should be serialized as CPU tensors."""
        original = torch.randn(3, 3).cuda()
        serialized = serialize_value(original)
        deserialized = deserialize_value(serialized)

        assert deserialized.device.type == "cpu"
        assert torch.equal(deserialized, original.cpu())
