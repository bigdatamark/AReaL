# [MODULE DOC]
# Dynamic-filtering predicate for SWE RL rollout groups: rejects groups whose rewards
# are
# all identical (no learning signal) and records rejection statistics.
from areal.utils import stats_tracker


def filter_function(sample):
    # [FUNCTION DOC]
    # Decides whether a rollout group is kept. Input: sample (dict) with "rewards" and
    # optionally "original_rewards" (preferred if present, i.e. pre-shaping rewards);
    # each is
    # an indexable sequence of numbers. Returns True iff not all rewards equal the
    # first.
    # Side effects: logs scalars to the "rollout" stats_tracker scope:
    # rejected_by_failed_or_perfect (1 if rejected else 0), and on rejection
    # rejected_by_all_correct (1 if all rewards > 0) / rejected_by_all_wrong (the
    # complement); both are 0 when accepted.
    # KNOWN ISSUE: an empty rewards sequence raises IndexError on rewards[0].
    # Use original_rewards if available for filtering and statistics
    rewards = (
        sample["original_rewards"]
        if "original_rewards" in sample
        else sample["rewards"]
    )
    reward0 = rewards[0]
    accept = any(r != reward0 for r in rewards)

    # Track rejection statistics
    if not accept:
        stats_tracker.get("rollout").scalar(rejected_by_failed_or_perfect=1)
        # Distinguish all-correct vs all-wrong
        all_positive = all(r > 0 for r in rewards)
        stats_tracker.get("rollout").scalar(rejected_by_all_correct=int(all_positive))
        stats_tracker.get("rollout").scalar(rejected_by_all_wrong=int(not all_positive))
    else:
        stats_tracker.get("rollout").scalar(rejected_by_failed_or_perfect=0)
        stats_tracker.get("rollout").scalar(rejected_by_all_correct=0)
        stats_tracker.get("rollout").scalar(rejected_by_all_wrong=0)

    return accept
