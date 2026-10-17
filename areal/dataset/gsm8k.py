# [MODULE DOC]
# GSM8K grade-school math loaders (HuggingFace "main" config) for SFT (tokenized
# question+answer with loss mask) and RL (chat messages prompting a boxed final answer).
# SPDX-License-Identifier: Apache-2.0

from datasets import load_dataset


def get_gsm8k_sft_dataset(
    path: str,
    split: str,
    tokenizer,
    max_length: int | None = None,
):
    # [FUNCTION DOC]
    # Loads the split, tokenizes each sample as question + answer + eos into input_ids,
    # and builds loss_mask with 0 for the question-token prefix length and 1 for the
    # rest. Drops question/answer columns and, if max_length is set, filters out samples
    # whose input_ids exceed it. Returns the mapped datasets.Dataset.
    # KNOWN ISSUE: The mask assumes tokenizing the question alone yields an exact prefix
    # of the full tokenization, which may not hold at the boundary for BPE tokenizers.
    dataset = load_dataset(path=path, name="main", split=split)

    def process(sample):
        # [FUNCTION DOC]
        # HELPER: Per-sample map function returning input_ids and loss_mask as described
        # above.
        seq_token = tokenizer.encode(
            sample["question"] + sample["answer"] + tokenizer.eos_token
        )
        prompt_token = tokenizer.encode(sample["question"])
        loss_mask = [0] * len(prompt_token) + [1] * (len(seq_token) - len(prompt_token))
        return {"input_ids": seq_token, "loss_mask": loss_mask}

    dataset = dataset.map(process).remove_columns(["question", "answer"])

    if max_length is not None:
        # Filter out sequences longer than max_length
        dataset = dataset.filter(lambda x: len(x["input_ids"]) <= max_length)

    return dataset


def get_gsm8k_rl_dataset(
    path: str,
    split: str,
    tokenizer,
    max_length: int | None = None,
):
    # [FUNCTION DOC]
    # Loads the split and converts each sample into messages: a single user turn with
    # the question plus an instruction to put the final answer in \boxed{}. Removes the
    # question column (answer is kept for reward). If max_length is set, filters samples
    # whose tokenized user content exceeds it. Returns the Dataset.
    dataset = load_dataset(path=path, name="main", split=split)

    def process(sample):
        # [FUNCTION DOC]
        # HELPER: Builds the one-message user chat list for a sample.
        messages = [
            {
                "role": "user",
                "content": sample["question"]
                + "\nPlease put your final answer within \\boxed{}.",
            }
        ]
        return {"messages": messages}

    dataset = dataset.map(process).remove_columns(["question"])

    # Filter out sequences longer than max_length if tokenizer and max_length are provided
    if max_length is not None:

        def filter_length(sample):
            # [FUNCTION DOC]
            # HELPER: Returns True if the tokenized first message content length is <=
            # max_length (no chat template applied).
            # Tokenize the user content to check length
            content = sample["messages"][0]["content"]
            tokens = tokenizer.encode(content)
            return len(tokens) <= max_length

        dataset = dataset.filter(filter_length)

    return dataset
