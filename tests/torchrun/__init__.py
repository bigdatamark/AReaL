# [PACKAGE DOC]
# Package of standalone scripts launched via torchrun by pytest wrappers in tests/.
# Each script runs one distributed scenario (FSDP, Megatron, Ulysses, vocab parallel,
# locks, weight sync, ...) and typically reports success by assertions or by writing
# "Passed"/"Failed" to an output file read back by the wrapper.
"""Torchrun distributed test scripts."""
