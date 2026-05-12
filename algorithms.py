"""
Provider-agnostic core algorithms for a MAKER-style agent workflow.

The module demonstrates three ideas:
1. decompose a long task into small state transitions;
2. use first-to-ahead-by-k voting for each transition;
3. discard red-flagged model outputs before they can win a vote.

This public excerpt is intentionally standalone. It does not include API keys,
provider clients, generated configs, prompts, or local execution history.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Callable, Optional


ModelFn = Callable[[Any], str]
ParserFn = Callable[[str], Any]
StateParserFn = Callable[[str], Any]
RedFlagChecker = Callable[[str], bool]


def generate_solution(
    initial_state: Any,
    model: ModelFn,
    k: int,
    num_steps: int,
    parse_action: ParserFn,
    parse_next_state: StateParserFn,
    check_red_flags: RedFlagChecker,
) -> list[Any]:
    """Run a multi-step task using voting at every state transition."""
    actions: list[Any] = []
    current_state = initial_state

    for _ in range(num_steps):
        action, next_state = do_voting(
            current_state,
            model,
            k,
            parse_action,
            parse_next_state,
            check_red_flags,
        )
        actions.append(action)
        current_state = next_state

    return actions


def do_voting(
    state: Any,
    model: ModelFn,
    k: int,
    parse_action: ParserFn,
    parse_next_state: StateParserFn,
    check_red_flags: RedFlagChecker,
) -> tuple[Any, Any]:
    """Sample until one action is ahead of every competitor by at least k votes."""
    if k < 1:
        raise ValueError("k must be at least 1")

    vote_counts: defaultdict[Any, int] = defaultdict(int)

    while True:
        action, next_state = get_vote(
            state,
            model,
            parse_action,
            parse_next_state,
            check_red_flags,
        )

        action_key = _to_hashable(action)
        vote_counts[action_key] += 1

        ordered_votes = sorted(vote_counts.values(), reverse=True)
        max_votes = ordered_votes[0]
        second_max_votes = ordered_votes[1] if len(ordered_votes) > 1 else 0

        if max_votes >= second_max_votes + k:
            for candidate_action, votes in vote_counts.items():
                if votes == max_votes:
                    return _from_hashable(candidate_action), next_state


def get_vote(
    state: Any,
    model: ModelFn,
    parse_action: ParserFn,
    parse_next_state: StateParserFn,
    check_red_flags: RedFlagChecker,
) -> tuple[Any, Any]:
    """Return one valid parsed model sample, resampling while red flags appear."""
    while True:
        response = model(state)
        if check_red_flags(response):
            continue

        action = parse_action(response)
        next_state = parse_next_state(response)
        return action, next_state


def create_red_flag_checker(
    max_words: Optional[int] = None,
    required_format_validator: Optional[Callable[[str], bool]] = None,
) -> RedFlagChecker:
    """Build a reusable checker for length and format violations."""

    def check_red_flags(response: str) -> bool:
        if max_words is not None and len(response.split()) > max_words:
            return True

        if required_format_validator is not None:
            return not required_format_validator(response)

        return False

    return check_red_flags


def estimate_k_min(
    num_steps: int,
    per_step_success_rate: float,
    target_success_rate: float = 0.9,
) -> int:
    """Estimate the minimum k needed to reach an overall target success rate."""
    if num_steps < 1:
        raise ValueError("num_steps must be at least 1")

    if not 0 < per_step_success_rate < 1:
        raise ValueError("per_step_success_rate must be between 0 and 1")

    if not 0 < target_success_rate < 1:
        raise ValueError("target_success_rate must be between 0 and 1")

    numerator = math.log(target_success_rate ** (-1 / num_steps) - 1)
    denominator = math.log((1 - per_step_success_rate) / per_step_success_rate)
    return math.ceil(numerator / denominator)


def _to_hashable(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_to_hashable(item) for item in value)

    if isinstance(value, dict):
        return tuple(sorted((key, _to_hashable(item)) for key, item in value.items()))

    return value


def _from_hashable(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_from_hashable(item) for item in value]

    return value
