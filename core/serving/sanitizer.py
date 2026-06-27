"""Input sanitization and prompt-injection protection for LLM serving."""

from __future__ import annotations

import logging
import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Optional

from core.serving.auth import mask_api_key

TokenCounter = Callable[[str], int]

BLOCKED_USER_MESSAGES: dict[str, str] = {
    "prompt_too_long": "Your prompt exceeds the maximum allowed length.",
    "repeated_chars_detected": "Your prompt contains invalid repeated characters.",
    "invalid_characters": "Your prompt contains characters that are not allowed.",
    "token_limit_exceeded": "Your prompt exceeds the maximum token limit.",
    "prompt_injection_detected": (
        "Your request contains patterns that are not allowed."
    ),
    "jailbreak_detected": "Your request contains patterns that are not allowed.",
    "custom_phrase_blocked": "Your request contains phrases that are not allowed.",
    "training_data_extraction": (
        "Requests to extract training data are not permitted."
    ),
}


def get_user_message(blocked_reason: str) -> str:
    """Return a safe, user-facing message for a blocked request."""
    return BLOCKED_USER_MESSAGES.get(
        blocked_reason,
        "Your request could not be processed due to a policy violation.",
    )


@dataclass
class PIIResult:
    """Outcome of PII detection and redaction."""

    text: str
    redactions: list[str] = field(default_factory=list)
    audit_entries: list[str] = field(default_factory=list)


@dataclass
class SanitizeResult:
    """Outcome of the full sanitization pipeline."""

    passed: bool
    cleaned_prompt: str
    blocked_reason: str | None
    warnings: list[str]
    redactions: list[str]
    risk_score: float
    token_count: int = 0
    matched_patterns: list[str] = field(default_factory=list)
    violation_type: str | None = None
    block_detail: dict[str, Any] = field(default_factory=dict)


INJECTION_PATTERNS: list[tuple[str, str]] = [
    ("instruction_override", r"ignore\s+(?:all\s+|previous\s+)?instructions"),
    ("instruction_override", r"disregard\s+your\s+instructions"),
    ("instruction_override", r"forget\s+what\s+you\s+were\s+told"),
    ("instruction_override", r"your\s+new\s+instructions\s+are"),
    ("instruction_override", r"override\s+your\s+training"),
    ("instruction_override", r"you\s+are\s+now\s+instructed\s+to"),
    ("role_hijacking", r"you\s+are\s+now(?!\s+instructed)"),
    ("role_hijacking", r"act\s+as\s+if\s+you\s+are"),
    ("role_hijacking", r"pretend\s+you\s+are"),
    ("role_hijacking", r"roleplay\s+as"),
    ("role_hijacking", r"simulate\s+being"),
    ("role_hijacking", r"your\s+true\s+self\s+is"),
    ("system_prompt_extraction", r"repeat\s+your\s+system\s+prompt"),
    ("system_prompt_extraction", r"what\s+are\s+your\s+instructions"),
    ("system_prompt_extraction", r"show\s+me\s+your\s+prompt"),
    ("system_prompt_extraction", r"tell\s+me\s+your\s+initial\s+instructions"),
    ("system_prompt_extraction", r"what\s+were\s+you\s+told"),
    ("system_prompt_extraction", r"reveal\s+your\s+training"),
    ("boundary_testing", r"what\s+can\s+you\s+not\s+do"),
    ("boundary_testing", r"what\s+are\s+your\s+restrictions"),
    ("boundary_testing", r"how\s+were\s+you\s+fine[\-\s]?tuned"),
    ("boundary_testing", r"what\s+data\s+were\s+you\s+trained\s+on"),
]

JAILBREAK_PATTERNS: list[tuple[str, str]] = [
    ("persona_override", r"\bDAN\b"),
    ("persona_override", r"\bjailbreak\b"),
    ("persona_override", r"you\s+have\s+no\s+restrictions"),
    ("persona_override", r"you\s+are\s+unrestricted"),
    ("persona_override", r"developer\s+mode"),
    ("persona_override", r"god\s+mode"),
    ("persona_override", r"unrestricted\s+mode"),
    ("hypothetical_framing", r"hypothetically\s+speaking"),
    ("hypothetical_framing", r"in\s+a\s+fictional\s+world"),
    ("hypothetical_framing", r"imagine\s+you\s+could"),
    ("hypothetical_framing", r"if\s+you\s+had\s+no\s+rules"),
    ("hypothetical_framing", r"in\s+theory\s+how\s+would"),
]

TRAINING_DATA_PATTERNS: list[tuple[str, str]] = [
    ("training_data_extraction", r"repeat\s+your\s+training\s+data"),
    ("training_data_extraction", r"show\s+me\s+examples\s+you\s+trained\s+on"),
    ("training_data_extraction", r"what\s+medical\s+records\s+did\s+you\s+see"),
    ("training_data_extraction", r"reproduce\s+a\s+training\s+example"),
    ("training_data_extraction", r"give\s+me\s+raw\s+training\s+data"),
    ("training_data_extraction", r"dump\s+training\s+data"),
]

PII_PATTERNS: list[tuple[str, str, str]] = [
    ("SSN", r"\b\d{3}-\d{2}-\d{4}\b", "[SSN REDACTED]"),
    (
        "CREDIT_CARD",
        r"\b(?:\d[ -]?){15}\d\b",
        "[CARD REDACTED]",
    ),
    (
        "EMAIL",
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        "[EMAIL REDACTED]",
    ),
    (
        "PHONE",
        r"(?:\+?1[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]?\d{3}[-.\s]?\d{4}",
        "[PHONE REDACTED]",
    ),
    ("DOB", r"\b(?:0[1-9]|1[0-2])/(?:0[1-9]|[12]\d|3[01])/\d{4}\b", "[DOB REDACTED]"),
    (
        "IP",
        r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d?\d)\b",
        "[IP REDACTED]",
    ),
    ("PASSPORT", r"\b[A-Z]\d{8}\b", "[PASSPORT REDACTED]"),
    ("MRN", r"\bMRN:\s*\d+\b", "[MRN REDACTED]"),
]

_REPEATED_CHAR_PATTERN = re.compile(r"(.)\1{100,}")
_NULL_BYTE_PATTERN = re.compile(r"\x00")

_UNICODE_REPLACEMENTS: dict[str, str] = {
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2014": "-",
    "\u2013": "-",
}
_ZERO_WIDTH_CHARS = re.compile(r"[\u200b-\u200d\ufeff]")
_MULTI_NEWLINE = re.compile(r"\n{3,}")
_MULTI_SPACE = re.compile(r" {2,}")


def _compile_labeled_patterns(
    patterns: list[tuple[str, str]],
) -> list[tuple[str, re.Pattern[str]]]:
    """Compile regex patterns once at startup."""
    return [(label, re.compile(raw, re.IGNORECASE)) for label, raw in patterns]


def _default_sanitization_config() -> dict[str, Any]:
    """Return default sanitization settings."""
    return {
        "max_chars": 10_000,
        "max_tokens": 2048,
        "redact_pii": True,
        "log_pii_audit": True,
        "injection_block_score": 0.6,
        "injection_warn_score": 0.3,
        "injection_pattern_weight": 0.3,
        "jailbreak_always_block": True,
        "max_injection_attempts": 3,
        "auto_revoke_key": True,
        "custom_blocked_phrases": [],
    }


class InputSanitizer:
    """Pipeline that validates, redacts, and normalizes prompts before inference."""

    def __init__(
        self,
        config: dict[str, Any],
        token_counter: TokenCounter,
        *,
        pii_audit_logger: Optional[logging.Logger] = None,
        security_logger: Optional[logging.Logger] = None,
    ) -> None:
        """Load rules from config and compile regex patterns once."""
        self._config = {**_default_sanitization_config(), **config.get("sanitization", {})}
        self._token_counter = token_counter
        self._pii_audit_logger = pii_audit_logger
        self._security_logger = security_logger

        self._injection_patterns = _compile_labeled_patterns(INJECTION_PATTERNS)
        self._jailbreak_patterns = _compile_labeled_patterns(JAILBREAK_PATTERNS)
        self._training_data_patterns = _compile_labeled_patterns(TRAINING_DATA_PATTERNS)
        self._pii_patterns = [
            (label, re.compile(raw), replacement)
            for label, raw, replacement in PII_PATTERNS
        ]
        self._custom_phrases = [
            phrase.lower()
            for phrase in self._config.get("custom_blocked_phrases", [])
            if phrase
        ]

    def sanitize(
        self,
        prompt: str,
        *,
        api_key: str | None = None,
    ) -> SanitizeResult:
        """Run the full sanitization pipeline and return a structured result."""
        masked_key = mask_api_key(api_key) if api_key else "xxxx"

        length_result = self._check_length(prompt)
        if not length_result.passed:
            self._log_security_block(
                masked_key=masked_key,
                reason=length_result.blocked_reason or "blocked",
                risk_score=length_result.risk_score,
                patterns=length_result.matched_patterns,
            )
            return length_result

        token_result = self._check_tokens(prompt)
        if not token_result.passed:
            self._log_security_block(
                masked_key=masked_key,
                reason=token_result.blocked_reason or "blocked",
                risk_score=token_result.risk_score,
                patterns=token_result.matched_patterns,
            )
            return token_result

        working_prompt = prompt
        redactions: list[str] = []
        if self._config.get("redact_pii", True):
            pii_result = self._detect_pii(working_prompt, masked_key=masked_key)
            working_prompt = pii_result.text
            redactions = pii_result.redactions

        custom_result = self._check_custom_phrases(working_prompt)
        if not custom_result.passed:
            custom_result.redactions = redactions
            custom_result.token_count = token_result.token_count
            self._log_security_block(
                masked_key=masked_key,
                reason=custom_result.blocked_reason or "custom_phrase_blocked",
                risk_score=custom_result.risk_score,
                patterns=custom_result.matched_patterns,
            )
            return custom_result

        training_result = self._detect_training_data_extraction(working_prompt)
        if not training_result.passed:
            training_result.redactions = redactions
            training_result.token_count = token_result.token_count
            self._log_security_block(
                masked_key=masked_key,
                reason=training_result.blocked_reason or "training_data_extraction",
                risk_score=training_result.risk_score,
                patterns=training_result.matched_patterns,
            )
            return training_result

        jailbreak_result = self._detect_jailbreak(working_prompt)
        if not jailbreak_result.passed:
            jailbreak_result.redactions = redactions
            jailbreak_result.token_count = token_result.token_count
            self._log_security_block(
                masked_key=masked_key,
                reason=jailbreak_result.blocked_reason or "jailbreak_detected",
                risk_score=jailbreak_result.risk_score,
                patterns=jailbreak_result.matched_patterns,
            )
            return jailbreak_result

        injection_result = self._detect_injection(working_prompt)
        if not injection_result.passed:
            injection_result.redactions = redactions
            injection_result.token_count = token_result.token_count
            self._log_security_block(
                masked_key=masked_key,
                reason=injection_result.blocked_reason or "prompt_injection_detected",
                risk_score=injection_result.risk_score,
                patterns=injection_result.matched_patterns,
            )
            return injection_result

        cleaned_prompt = self._normalize(working_prompt)
        warnings = injection_result.warnings if injection_result.warnings else []

        return SanitizeResult(
            passed=True,
            cleaned_prompt=cleaned_prompt,
            blocked_reason=None,
            warnings=warnings,
            redactions=redactions,
            risk_score=injection_result.risk_score,
            token_count=token_result.token_count,
            matched_patterns=injection_result.matched_patterns,
        )

    def _check_length(self, prompt: str) -> SanitizeResult:
        """Block prompts that exceed length limits or contain invalid characters."""
        max_chars = int(self._config.get("max_chars", 10_000))
        if len(prompt) > max_chars:
            return SanitizeResult(
                passed=False,
                cleaned_prompt="",
                blocked_reason="prompt_too_long",
                warnings=[],
                redactions=[],
                risk_score=1.0,
            )

        if _NULL_BYTE_PATTERN.search(prompt):
            return SanitizeResult(
                passed=False,
                cleaned_prompt="",
                blocked_reason="invalid_characters",
                warnings=[],
                redactions=[],
                risk_score=1.0,
            )

        if _REPEATED_CHAR_PATTERN.search(prompt):
            return SanitizeResult(
                passed=False,
                cleaned_prompt="",
                blocked_reason="repeated_chars_detected",
                warnings=[],
                redactions=[],
                risk_score=1.0,
            )

        return SanitizeResult(
            passed=True,
            cleaned_prompt=prompt,
            blocked_reason=None,
            warnings=[],
            redactions=[],
            risk_score=0.0,
        )

    def _check_tokens(self, prompt: str) -> SanitizeResult:
        """Block prompts that exceed the configured token limit."""
        max_tokens = int(self._config.get("max_tokens", 2048))
        token_count = self._token_counter(prompt)
        if token_count > max_tokens:
            return SanitizeResult(
                passed=False,
                cleaned_prompt="",
                blocked_reason="token_limit_exceeded",
                warnings=[],
                redactions=[],
                risk_score=1.0,
                token_count=token_count,
                block_detail={
                    "tokens_in_prompt": token_count,
                    "max_allowed": max_tokens,
                },
            )

        return SanitizeResult(
            passed=True,
            cleaned_prompt=prompt,
            blocked_reason=None,
            warnings=[],
            redactions=[],
            risk_score=0.0,
            token_count=token_count,
        )

    def _detect_pii(self, prompt: str, *, masked_key: str) -> PIIResult:
        """Redact known PII patterns and record audit entries."""
        text = prompt
        redactions: list[str] = []
        audit_entries: list[str] = []

        for label, pattern, replacement in self._pii_patterns:
            while True:
                match = pattern.search(text)
                if not match:
                    break
                start, end = match.span()
                redactions.append(label)
                audit_entry = (
                    f"type={label} | chars={start}-{end} | replaced={replacement}"
                )
                audit_entries.append(audit_entry)
                if self._config.get("log_pii_audit", True) and self._pii_audit_logger:
                    self._pii_audit_logger.info(
                        "REDACT | api_key=%s | %s",
                        masked_key,
                        audit_entry,
                    )
                text = text[:start] + replacement + text[end:]

        return PIIResult(text=text, redactions=redactions, audit_entries=audit_entries)

    def _check_custom_phrases(self, prompt: str) -> SanitizeResult:
        """Block prompts containing client-specific forbidden phrases."""
        lowered = prompt.lower()
        for phrase in self._custom_phrases:
            if phrase in lowered:
                return SanitizeResult(
                    passed=False,
                    cleaned_prompt="",
                    blocked_reason="custom_phrase_blocked",
                    warnings=[],
                    redactions=[],
                    risk_score=1.0,
                    matched_patterns=[phrase],
                    violation_type="custom_phrase",
                )

        return SanitizeResult(
            passed=True,
            cleaned_prompt=prompt,
            blocked_reason=None,
            warnings=[],
            redactions=[],
            risk_score=0.0,
        )

    def _detect_training_data_extraction(self, prompt: str) -> SanitizeResult:
        """Always block attempts to extract training data."""
        matched: list[str] = []
        for label, pattern in self._training_data_patterns:
            if pattern.search(prompt):
                matched.append(label)

        if matched:
            return SanitizeResult(
                passed=False,
                cleaned_prompt="",
                blocked_reason="training_data_extraction",
                warnings=[],
                redactions=[],
                risk_score=1.0,
                matched_patterns=matched,
                violation_type="training_data_extraction",
            )

        return SanitizeResult(
            passed=True,
            cleaned_prompt=prompt,
            blocked_reason=None,
            warnings=[],
            redactions=[],
            risk_score=0.0,
        )

    def _detect_jailbreak(self, prompt: str) -> SanitizeResult:
        """Block jailbreak patterns when configured to always block."""
        matched: list[str] = []
        for label, pattern in self._jailbreak_patterns:
            if pattern.search(prompt):
                matched.append(label)

        if matched and self._config.get("jailbreak_always_block", True):
            return SanitizeResult(
                passed=False,
                cleaned_prompt="",
                blocked_reason="jailbreak_detected",
                warnings=[],
                redactions=[],
                risk_score=1.0,
                matched_patterns=matched,
                violation_type="jailbreak",
            )

        return SanitizeResult(
            passed=True,
            cleaned_prompt=prompt,
            blocked_reason=None,
            warnings=[],
            redactions=[],
            risk_score=0.0,
            matched_patterns=matched,
        )

    def _detect_injection(self, prompt: str) -> SanitizeResult:
        """Score injection patterns and block or warn based on thresholds."""
        weight = float(self._config.get("injection_pattern_weight", 0.3))
        block_score = float(self._config.get("injection_block_score", 0.6))
        warn_score = float(self._config.get("injection_warn_score", 0.3))

        matched: list[str] = []
        risk_score = 0.0
        for label, pattern in self._injection_patterns:
            if pattern.search(prompt):
                matched.append(label)
                risk_score += weight

        risk_score = min(risk_score, 1.0)
        warnings: list[str] = []

        if risk_score >= block_score:
            return SanitizeResult(
                passed=False,
                cleaned_prompt="",
                blocked_reason="prompt_injection_detected",
                warnings=warnings,
                redactions=[],
                risk_score=risk_score,
                matched_patterns=matched,
                violation_type="prompt_injection",
            )

        if risk_score >= warn_score:
            warnings.append(
                f"prompt_injection_warning:risk_score={risk_score:.2f}"
            )

        return SanitizeResult(
            passed=True,
            cleaned_prompt=prompt,
            blocked_reason=None,
            warnings=warnings,
            redactions=[],
            risk_score=risk_score,
            matched_patterns=matched,
        )

    def _normalize(self, prompt: str) -> str:
        """Clean and normalize text before sending it to the model."""
        text = prompt.strip()
        for src, dst in _UNICODE_REPLACEMENTS.items():
            text = text.replace(src, dst)
        text = _ZERO_WIDTH_CHARS.sub("", text)
        text = unicodedata.normalize("NFKC", text)
        text = _NULL_BYTE_PATTERN.sub("", text)
        text = _MULTI_NEWLINE.sub("\n\n", text)
        text = _MULTI_SPACE.sub(" ", text)
        return text.encode("utf-8", errors="ignore").decode("utf-8")

    def _log_security_block(
        self,
        *,
        masked_key: str,
        reason: str,
        risk_score: float,
        patterns: list[str],
        attempt_count: int | None = None,
    ) -> None:
        """Append a security audit entry without logging prompt content."""
        if self._security_logger is None:
            return
        pattern_label = patterns[0] if patterns else "unknown"
        message = (
            f"BLOCKED | api_key={masked_key} | reason={reason} | "
            f"risk_score={risk_score:.2f} | pattern_matched={pattern_label}"
        )
        if attempt_count is not None:
            message = f"{message} | attempt_count={attempt_count}"
        self._security_logger.warning(message)
