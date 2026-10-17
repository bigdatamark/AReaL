# [MODULE DOC]
# Anthropic HH-RLHF preference dataset loaders for reward-model training
# (chosen/rejected token IDs) and DPO (token IDs plus response loss masks).
# SPDX-License-Identifier: Apache-2.0

from datasets import load_dataset


def get_hhrlhf_rw_dataset(
    path: str,
    split: str,
    tokenizer,
    max_length: int | None = None,
):
    # [FUNCTION DOC]
    # Loads the split and tokenizes chosen and rejected texts (each with eos appended)
    # into chosen_ids/rejected_ids, dropping the text columns. If max_length is set
    # keeps only pairs where both sequences fit. Returns the Dataset.
    dataset = load_dataset(path=path, split=split)

    def process(sample):
        # [FUNCTION DOC]
        # HELPER: Per-sample map producing chosen_ids and rejected_ids.
        chosen_seq_token = tokenizer.encode(sample["chosen"] + tokenizer.eos_token)
        rejected_seq_token = tokenizer.encode(sample["rejected"] + tokenizer.eos_token)
        return {"chosen_ids": chosen_seq_token, "rejected_ids": rejected_seq_token}

    dataset = dataset.map(process).remove_columns(["chosen", "rejected"])

    if max_length is not None:
        # Filter out sequences longer than max_length
        dataset = dataset.filter(
            lambda x: (
                (len(x["chosen_ids"]) <= max_length)
                and (len(x["rejected_ids"]) <= max_length)
            )
        )

    return dataset


def get_hhrlhf_dpo_dataset(
    path: str,
    split: str,
    tokenizer,
    max_length: int | None = None,
):
    # [FUNCTION DOC]
    # Loads the split and per sample tokenizes chosen/rejected (with eos), takes the
    # shared prompt length as the longest common token prefix, and emits ids plus loss
    # masks (0 over the shared prefix, 1 over each response). Drops text columns; if
    # max_length is set keeps pairs where both sequences fit. Returns the Dataset;
    # reference logprobs are computed online later.
    # KNOWN ISSUE: Docstring calls the masks boolean but they are lists of 0/1 ints. The
    # common-prefix heuristic can include identical leading response tokens in the
    # "prompt" region.
    """Load HH-RLHF dataset for DPO training.

    Each sample will contain:
    - ``chosen_ids`` / ``rejected_ids``: full token ids (prompt + response).
    - ``chosen_loss_mask`` / ``rejected_loss_mask``: boolean mask where ``True``
      marks the response tokens that participate in the loss.

    Reference log-probabilities are computed online by the ref engine during
    training (configured via the ``ref`` field in ``DPOConfig``).
    """
    dataset = load_dataset(path=path, split=split)

    def process(sample):
        # [FUNCTION DOC]
        # HELPER: Computes chosen/rejected ids, common-prefix length and the two loss
        # masks for one sample.
        chosen_ids = tokenizer.encode(sample["chosen"] + tokenizer.eos_token)
        rejected_ids = tokenizer.encode(sample["rejected"] + tokenizer.eos_token)

        prompt_len = 0
        for c, r in zip(chosen_ids, rejected_ids):
            if c == r:
                prompt_len += 1
            else:
                break

        return {
            "chosen_ids": chosen_ids,
            "rejected_ids": rejected_ids,
            "chosen_loss_mask": [0] * prompt_len + [1] * (len(chosen_ids) - prompt_len),
            "rejected_loss_mask": [0] * prompt_len
            + [1] * (len(rejected_ids) - prompt_len),
        }

    dataset = dataset.map(process).remove_columns(["chosen", "rejected"])

    if max_length is not None:
        dataset = dataset.filter(
            lambda x: (len(x["chosen_ids"]) <= max_length)
            and (len(x["rejected_ids"]) <= max_length)
        )

    return dataset
