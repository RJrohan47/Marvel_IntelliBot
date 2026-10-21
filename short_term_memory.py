"""Keep model requests small while the checkpointer keeps the full conversation."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from time import perf_counter

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, messages_to_dict
from langchain_core.messages.utils import count_tokens_approximately, trim_messages

from model_fallback import invoke_with_fallback

logger = logging.getLogger(__name__)

# Shared soft budget: keep it below the smallest configured model's context window.
# The counter is approximate, so leave room for provider overhead and the answer.
CONTEXT_TOKEN_BUDGET = int(os.getenv("STM_CONTEXT_TOKEN_BUDGET", "24000"))
RECENT_TOKEN_BUDGET = int(os.getenv("STM_RECENT_TOKEN_BUDGET", "8000"))
SUMMARY_BATCH_TOKEN_BUDGET = int(os.getenv("STM_SUMMARY_BATCH_TOKEN_BUDGET", "8000"))
SUMMARY_TARGET_TOKEN_BUDGET = int(os.getenv("STM_SUMMARY_TARGET_TOKEN_BUDGET", "1500"))
CHARS_PER_TOKEN = 3.5  # Slightly conservative because this is an estimate.


@dataclass
class ModelContext:
    messages: list[BaseMessage]
    summary: str
    summarized_message_count: int


def _count(messages: list[BaseMessage], tools=None) -> int:
    """Estimate request size, including the tool schemas when supplied."""
    return count_tokens_approximately(
        messages, chars_per_token=CHARS_PER_TOKEN, tools=tools
    )


def _summary_prompt(summary: str) -> SystemMessage:
    preserve = (
        "Preserve exact names, dates, amounts, identifiers, URLs, decisions, and unresolved "
        "questions, along with the user's goals and preferences. Distinguish stated facts from "
        "assumptions, and do not invent details. "
    )
    if summary:
        content = (
            "Use the conversation summary below as background. Update it with the new messages. "
            f"{preserve}"
            f"Keep the result concise (about {SUMMARY_TARGET_TOKEN_BUDGET} tokens).\n\n"
            f"Existing summary:\n{summary}"
        )
    else:
        content = (
            "Summarize the conversation messages that follow for use as short-term context in "
            f"later turns. {preserve}Keep it concise (about {SUMMARY_TARGET_TOKEN_BUDGET} tokens)."
        )
    return SystemMessage(content=content)


def _model_messages(
    system_message: SystemMessage,
    summary: str,
    history: list[BaseMessage],
    *,
    context_note: str = "",
) -> list[BaseMessage]:
    """Build the main model's prompt from the system message, summary, and history."""
    messages: list[BaseMessage] = [system_message]
    if context_note:
        messages.append(SystemMessage(content=context_note))
    if summary:
        messages.append(SystemMessage(content=f"Conversation summary for background:\n{summary}"))
    messages.extend(history)
    return messages


def _complete_turn_batches(messages: list[BaseMessage]):
    """Yield oldest whole turns, stopping before the batch token budget is exceeded."""
    turns: list[list[BaseMessage]] = []
    for message in messages:
        if isinstance(message, HumanMessage) and turns:
            turns.append([])
        elif not turns:
            turns.append([])
        turns[-1].append(message)

    batch: list[BaseMessage] = []
    for turn in turns:
        if batch and _count(batch + turn) > SUMMARY_BATCH_TOKEN_BUDGET:
            yield batch
            batch = []
        batch.extend(turn)
        # Include a single unusually large turn so compaction can still make progress.
        if _count(batch) >= SUMMARY_BATCH_TOKEN_BUDGET:
            yield batch
            batch = []
    if batch:
        yield batch


def _response_text(response) -> str:
    """Convert a model response to plain summary text."""
    content = response.content
    if isinstance(content, list):
        content = " ".join(
            str(part.get("text", "")) if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content).strip()


def _summarize(summary_model, fallback_summary_model, summary: str, messages: list[BaseMessage]) -> str:
    request = [
        _summary_prompt(summary),
        HumanMessage(content=json.dumps(messages_to_dict(messages), ensure_ascii=False, default=str)),
    ]
    started = perf_counter()
    request_tokens = _count(request)
    result = _response_text(
        invoke_with_fallback(summary_model, fallback_summary_model, request)
    )

    # If the model ignores the requested size, ask once for a shorter version.
    shortened = False
    if _count([HumanMessage(content=result)]) > SUMMARY_TARGET_TOKEN_BUDGET:
        shortened = True
        shorter_request = [
            SystemMessage(
                content=(
                    f"Shorten this summary to about {SUMMARY_TARGET_TOKEN_BUDGET} tokens. "
                    "Preserve exact names, dates, amounts, identifiers, URLs, decisions, and "
                    "open questions. Do not add facts."
                )
            ),
            HumanMessage(content=result),
        ]
        result = _response_text(
            invoke_with_fallback(summary_model, fallback_summary_model, shorter_request)
        )

    summary_tokens = _count([HumanMessage(content=result)])
    logger.info(
        "STM summary completed: input_tokens~%s summary_tokens~%s shortened=%s duration_ms=%s",
        request_tokens,
        summary_tokens,
        shortened,
        round((perf_counter() - started) * 1000),
    )
    if summary_tokens > SUMMARY_TARGET_TOKEN_BUDGET:
        raise ValueError("summary remained above its approximate token target")
    return result


def _latest_turn(history: list[BaseMessage]) -> list[BaseMessage]:
    """Keep the latest user turn and its assistant/tool messages together."""
    latest_user = next(
        (index for index in range(len(history) - 1, -1, -1)
         if isinstance(history[index], HumanMessage)),
        None,
    )
    return history[latest_user:] if latest_user is not None else history[-1:]


def prepare_model_context(
    *,
    system_message: SystemMessage,
    history: list[BaseMessage],
    summary: str = "",
    summarized_message_count: int = 0,
    tools=None,
    summary_model,
    fallback_summary_model=None,
) -> ModelContext:
    """Summarize older turns and pass only the summary plus recent turns to the model.

    `history` and the checkpoint are never changed. The returned message count marks
    the prefix represented by the summary, so the same old messages are not summarized
    again on the next turn.
    """
    started = perf_counter()
    summary = summary or ""
    summarized_message_count = min(max(summarized_message_count, 0), len(history))

    def finish(prompt: list[BaseMessage], stage: str) -> ModelContext:
        estimated_tokens = _count(prompt, tools)
        logger.info(
            "STM context ready: stage=%s estimated_tokens~%s budget=%s "
            "summary_tokens~%s summarized_messages=%s prompt_messages=%s duration_ms=%s",
            stage,
            estimated_tokens,
            CONTEXT_TOKEN_BUDGET,
            _count([HumanMessage(content=summary)]) if summary else 0,
            summarized_message_count,
            len(prompt),
            round((perf_counter() - started) * 1000),
        )
        if estimated_tokens > CONTEXT_TOKEN_BUDGET:
            logger.warning(
                "STM minimum context is above its soft budget: estimated_tokens~%s budget=%s",
                estimated_tokens,
                CONTEXT_TOKEN_BUDGET,
            )
        return ModelContext(prompt, summary, summarized_message_count)

    while True:
        # The saved summary stands in for this prefix of the full checkpointed history.
        active_history = history[summarized_message_count:]
        prompt = _model_messages(system_message, summary, active_history)
        if _count(prompt, tools) <= CONTEXT_TOKEN_BUDGET:
            return finish(prompt, "within_budget")

        # Keep recent complete turns, with room left for prompts, tools, and the response.
        prompt_overhead = _count(_model_messages(system_message, summary, []), tools)
        recent_budget = min(
            RECENT_TOKEN_BUDGET,
            max(1, CONTEXT_TOKEN_BUDGET - prompt_overhead - 1000),
        )
        
        recent_history = trim_messages(
            active_history,
            max_tokens=recent_budget,
            token_counter=lambda messages: _count(messages),
            strategy="last",
            allow_partial=False,
            start_on="human",
        )
        if not recent_history:
            logger.warning("STM could not fit a recent turn; using the latest turn and saved summary")
            latest_prompt = _model_messages(
                system_message,
                summary,
                _latest_turn(active_history),
                context_note=(
                    "Older conversation was omitted because it could not be compacted for this request. "
                    "Use only the available summary and messages. Do not guess missing details; "
                    "if a missing detail is needed, ask the user a focused follow-up question."
                ),
            )
            return finish(latest_prompt, "latest_turn_fallback")

        # Messages before the first retained turn are candidates for summarization.
        first_recent_index = next(
            (i for i, message in enumerate(active_history) if message is recent_history[0]),
            None,
        )
        if first_recent_index is None or first_recent_index == 0:
            logger.warning("STM found no older complete turns to summarize")
            recent_prompt = _model_messages(system_message, summary, recent_history)
            return finish(recent_prompt, "recent_turns_only")

        older_history = active_history[:first_recent_index]
        oldest_batch = next(iter(_complete_turn_batches(older_history)), [])
        if not oldest_batch:
            recent_prompt = _model_messages(system_message, summary, recent_history)
            return finish(recent_prompt, "recent_turns_only")

        try:
            updated_summary = _summarize(
                summary_model, fallback_summary_model, summary, oldest_batch
            )
            if not updated_summary:
                raise ValueError("summary model returned empty text")
        except Exception:
            logger.exception("STM summarization failed; using the previous summary and recent turns")
            recent_prompt = _model_messages(
                system_message,
                summary,
                recent_history,
                context_note=(
                    "Some older conversation could not be summarized for this request. "
                    "Do not assume missing details; if they matter, ask the user to restate them."
                ),
            )
            return finish(recent_prompt, "summary_failure_fallback")

        # Summarize the oldest batch, then repeat the size check before summarizing more.
        summary = updated_summary
        summarized_message_count += len(oldest_batch)
