# MAKER framework excerpt

Sanitized excerpt from a local implementation of the MAKER pattern: maximal agentic decomposition, first-to-ahead-by-k voting, and red-flag filtering for long-running LLM tasks.

The original local project includes an interactive CLI and provider integration. This public showcase keeps the provider-agnostic core algorithm so the design can be reviewed without exposing API keys, runtime configs, prompts, generated task files, or local execution history.

## What this demonstrates

- Breaking a large task into verifiable micro-steps.
- Sampling multiple model outputs for each step.
- Accepting a step only when one candidate is ahead by `k` votes.
- Rejecting responses that violate length or format constraints before they enter the vote.
- Estimating the voting parameter needed for a target success rate.

## Files

- [`algorithms.py`](./algorithms.py) - standalone voting, red-flagging, and `k_min` estimation helpers.

## Safety

This directory does not include API keys, `.env` files, generated configs, provider request logs, or private task data.
