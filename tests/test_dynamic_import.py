# [MODULE DOC]
# Unit tests for import_from_string (areal.utils.dynamic_import), which resolves a
# dotted "module.attr" path to an object. Covers valid classes/functions, invalid
# modules (ImportError), missing attributes (AttributeError) and malformed inputs
# (ValueError).
from unittest.mock import patch

import pytest

from areal.utils.dynamic_import import import_from_string


class TestImportFromString:
    # [CLASS DOC]
    # Groups the import_from_string test cases; no state.
    def test_import_class(self):
        # [FUNCTION DOC]
        # Imports RolloutWorkflow by path; asserts it is a class with the right name.
        WorkflowClass = import_from_string("areal.api.workflow_api.RolloutWorkflow")
        assert WorkflowClass.__name__ == "RolloutWorkflow"
        assert isinstance(WorkflowClass, type)

    def test_import_function(self):
        # [FUNCTION DOC]
        # Imports concat_padded_tensors; asserts it is callable with the right name.
        func = import_from_string("areal.utils.data.concat_padded_tensors")
        assert callable(func)
        assert func.__name__ == "concat_padded_tensors"

    def test_invalid_module(self):
        # [FUNCTION DOC]
        # Asserts a nonexistent module raises ImportError "Failed to import module".
        with pytest.raises(ImportError, match="Failed to import module"):
            import_from_string("nonexistent.module.SomeClass")

    def test_invalid_attribute(self):
        # [FUNCTION DOC]
        # Asserts a missing attribute on a real module raises AttributeError "has no
        # attribute".
        with pytest.raises(AttributeError, match="has no attribute"):
            import_from_string("areal.utils.data.nonexistent_function")

    def test_invalid_format_no_dot(self):
        # [FUNCTION DOC]
        # Asserts a path without dots raises ValueError "Invalid module path".
        with pytest.raises(ValueError, match="Invalid module path"):
            import_from_string("NoDotsHere")

    def test_empty_string(self):
        # [FUNCTION DOC]
        # Asserts the empty string raises ValueError "Invalid module path".
        with pytest.raises(ValueError, match="Invalid module path"):
            import_from_string("")

    def test_none_input(self):
        # [FUNCTION DOC]
        # Asserts None raises ValueError "Invalid module path".
        with pytest.raises(ValueError, match="Invalid module path"):
            import_from_string(None)

    def test_non_string_input(self):
        # [FUNCTION DOC]
        # Asserts int, list and dict inputs each raise ValueError "Invalid module path".
        """Test that non-string inputs raise ValueError"""
        with pytest.raises(ValueError, match="Invalid module path"):
            import_from_string(123)

        with pytest.raises(ValueError, match="Invalid module path"):
            import_from_string([])

        with pytest.raises(ValueError, match="Invalid module path"):
            import_from_string({})

    def test_multiple_dots_in_path(self):
        # [FUNCTION DOC]
        # Asserts deeply nested paths resolve (a function and MultiTurnWorkflow class).
        """Test module paths with multiple dots"""
        # Valid path with multiple dots
        obj = import_from_string("areal.utils.data.concat_padded_tensors")
        assert callable(obj)

        # Another valid nested path
        workflow_class = import_from_string(
            "areal.workflow.multi_turn.MultiTurnWorkflow"
        )
        assert isinstance(workflow_class, type)

    def test_import_with_circular_import(self):
        # [FUNCTION DOC]
        # Patches importlib.import_module to raise ImportError and asserts it is re-
        # raised as ImportError "Failed to import module".
        """Test handling of circular imports (should raise ImportError)"""
        # This test simulates a circular import scenario
        with patch("importlib.import_module") as mock_import:
            mock_import.side_effect = ImportError("circular import detected")

            with pytest.raises(ImportError, match="Failed to import module"):
                import_from_string("some.module.CircularClass")

    def test_import_module_with_side_effects(self):
        # [FUNCTION DOC]
        # Imports areal.utils.logging.getLogger (module with import-time side effects);
        # asserts callable with the right name.
        """Test that module import works even with side effects"""
        # Test importing a module that might have initialization code
        obj = import_from_string("areal.utils.logging.getLogger")
        assert callable(obj)
        assert obj.__name__ == "getLogger"

    def test_error_messages_are_descriptive(self):
        # [FUNCTION DOC]
        # Asserts ImportError messages mention the module and attribute names, and
        # AttributeError messages mention the module path and missing attribute.
        """Test that error messages provide helpful information"""
        # Test module not found error
        with pytest.raises(ImportError) as exc_info:
            import_from_string("nonexistent_module.SomeClass")

        error_msg = str(exc_info.value)
        assert "nonexistent_module" in error_msg
        assert "SomeClass" in error_msg

        # Test attribute not found error
        with pytest.raises(AttributeError) as exc_info:
            import_from_string("areal.utils.data.NonExistentFunction")

        error_msg = str(exc_info.value)
        assert "areal.utils.data" in error_msg
        assert "NonExistentFunction" in error_msg

    def test_import_builtin_function(self):
        # [FUNCTION DOC]
        # Imports os.path.join and asserts joining "a","b","c" yields the platform-
        # appropriate path.
        """Test importing built-in functions from standard library modules"""
        # This should work for any valid Python module path
        func = import_from_string("os.path.join")
        assert callable(func)
        # Verify it's the actual function by calling with correct string arguments
        result = func("a", "b", "c")
        assert result == "a/b/c" or result == "a\\b\\c"
