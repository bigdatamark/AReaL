# [MODULE DOC]
# Custom exception types distinguishing engine-side errors from framework-side errors,
# each carrying an error type, sub-error and reason.
# SPDX-License-Identifier: Apache-2.0


class EngineError(Exception):
    # [CLASS DOC]
    # Exception for inference/training engine failures.
    def __init__(self, error_type, sub_error, reason):
        # [FUNCTION DOC]
        # Stores error_type, sub_error, reason as attributes and formats the message
        # "error_type: ..., sub_error: ..., reason: ...".
        self.error_type = error_type
        self.sub_error = sub_error
        self.reason = reason
        super().__init__(
            f"error_type: {error_type}, sub_error: {sub_error}, reason: {reason}"
        )


class FrameworkError(Exception):
    # [CLASS DOC]
    # Exception for framework (AReaL-side) failures; same structure as EngineError.
    def __init__(self, error_type, sub_error, reason):
        # [FUNCTION DOC]
        # Stores error_type, sub_error, reason and formats the same message as
        # EngineError.
        self.error_type = error_type
        self.sub_error = sub_error
        self.reason = reason
        super().__init__(
            f"error_type: {error_type}, sub_error: {sub_error}, reason: {reason}"
        )
