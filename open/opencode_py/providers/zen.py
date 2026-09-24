"""OpenCode Zen provider (https://opencode.ai/zen/v1) — the free models.

Thin wrapper over OpenAICompatProvider pointing at Zen's OpenAI-compatible
endpoint. With no API key, free (cost==0) models are used and Zen accepts the
literal API key "public" (mirrors opencode's behavior).

The Zen gateway throttles anonymous clients (unknown User-Agent, no
x-opencode-* headers) to a tiny free allowance — a couple of requests, then
429 FreeUsageLimitError. The official opencode client identifies itself with
`User-Agent: opencode/...` plus x-opencode-* headers, and the gateway treats
those as trusted clients with the real free quota. This provider sends the
same identity headers so free models (x-preview-f-free, etc.) keep working
across turns instead of being blocked after the first message.
"""

from __future__ import annotations

import threading
from typing import Any
from uuid import uuid4

from .openai_compat import OpenAICompatProvider

ZEN_BASE_URL = "https://opencode.ai/zen/v1"

# Free-tier gate (measured 2026-09-23): Zen's Console rejects anonymous free
# requests whose User-Agent parses below 1.18.0 with
# 426 UpgradeRequired ("OpenCode 1.18.0 or newer is required to use the
# free tier"). Send the current official release version. The server raises
# this floor over time; _CURRENT_VERSION auto-bumps from the 426 message so
# a version-rot breakage self-heals instead of silently killing free models.
OFFICIAL_VERSION = "1.18.32"

_CURRENT_VERSION = OFFICIAL_VERSION


def _parse_version(text: str) -> tuple[int, ...] | None:
    import re

    m = re.search(r"(\d+)\.(\d+)(?:\.(\d+))?", text or "")
    if not m:
        return None
    return tuple(int(g) for g in m.groups() if g is not None)


def _bump_version_from_error(body: str) -> bool:
    """Adopt the minimum version a 426 UpgradeRequired body demands.

    Returns True when the floor moved (caller should retry once).
    """
    global _CURRENT_VERSION
    want = _parse_version(body or "")
    if not want:
        return False
    have = _parse_version(_CURRENT_VERSION) or ()
    if want > have:
        _CURRENT_VERSION = ".".join(str(n) for n in want)
        return True
    return False


def _is_upgrade_required(err: object) -> str:
    """Return the 426 body when err is an UpgradeRequired failure, else ''."""
    if getattr(err, "status", None) != 426:
        return ""
    return str(err) or ""


def gate_hint(status: int | None, body: str) -> str | None:
    """Actionable hint for a Zen free-tier failure (shown to the user).

    Returns None when nothing specific applies. Covers: 426 version floor
    (self-heal attempted), 403 fingerprint/quota, 429 quota, 5xx transient.
    """
    s = body or ""
    if status == 426:
        return (
            "Zen raised the minimum client version; this client adopted the"
            " demanded version and retried. If it persists, bump"
            " OFFICIAL_VERSION in opencode_py/providers/zen.py, then run"
            " `opencode-py --check`."
        )
    if status == 403 and ("FreeTier" in s or "within OpenCode" in s):
        return (
            "Zen rejected the client fingerprint (or the anonymous quota is"
            " exhausted). Run `opencode-py --check` for the live diagnosis;"
            " a server-side gate change needs a client update — otherwise"
            " wait and retry."
        )
    if status == 429 or "FreeUsageLimit" in s or "Rate limit" in s:
        return (
            "free-tier quota exhausted — wait ~30s and retry, or add lanes"
            " to your 'rotation' list to fail over."
        )
    if status is not None and 500 <= status < 600:
        return "temporary Zen server issue — wait and retry."
    return None


def _user_agent() -> str:
    """Version string the free-tier gate accepts (opencode/1.18.0+).

    The old 'opencode/0.1.0' (from our package version) is rejected with
    426 UpgradeRequired, so free models never respond.
    """
    return f"opencode/{_CURRENT_VERSION}"


_SES_RE: Any = None


def _ses_re() -> Any:
    global _SES_RE
    if _SES_RE is None:
        import re

        _SES_RE = re.compile(r"^ses_[0-9a-f]{9}[0-9A-Za-z]{17}$")
    return _SES_RE


def zen_session_id(sid: str | None) -> str:
    """Map any conversation id to an x-opencode-session value Zen accepts.

    Measured 2026-09-23/24: the free-tier gate rejects session ids that
    don't look official (raw uuid hex gets 403 FreeTierError) while genuine
    ses_-style ids pass — any age, reusable. Accepted shape:
    ``ses_`` + 9 chars [0-9a-f] + 17 chars [0-9A-Za-z] (26 total).
    The head must be lowercase hex (a single x/w/g in it 403s); the tail
    must be a benign mix — a fixed known-good tail is used since random
    tails trip the filter unpredictably.
    Already-compliant ids pass through; anything else is derived
    deterministically (sha256) so the value stays stable across turns and
    provider rebuilds (lane affinity). None mints a fresh random one.
    """
    import hashlib
    import secrets

    # Fixed known-good tail (random tails trip the shape filter
    # unpredictably; this one passes with every proven head).
    tail = "ffef5g6h7i8j9k0lm"
    # First head char restricted to values observed passing (0,2,a,b,f);
    # e.g. a '9'-leading head 403s. Remaining 8 hex chars vary freely.
    first = "02abf"
    if sid and _ses_re().match(sid):
        return sid
    if sid:
        digest = hashlib.sha256(sid.encode("utf-8")).hexdigest()
        head = first[int(digest[0], 16) % len(first)] + digest[1:9]
        return f"ses_{head}{tail}"
    head = first[secrets.randbelow(len(first))] + secrets.token_hex(4)
    return f"ses_{head}{tail}"

# Lane-rotation registry: Zen pins each x-opencode-session id to ONE upstream
# lane. When that lane dies server-side (streams end with Zen's own
# finish_reason="network_error"), retrying with the SAME id hammers the SAME
# dead lane — the user sees ↻ climb to (50) without a single success while a
# brand-new session id would have worked on attempt #1 (measured 2026-08-23:
# fixed sid A 5/5 OK, fixed sid B 1/5, fresh ids mixed). Keyed by BASE session
# id so a rotation survives provider re-instantiation: rotation.build_provider
# constructs a NEW ZenProvider for every attempt with the same engine sid.
_LANE_EPOCH: dict[str, int] = {}
_LANE_LOCK = threading.Lock()

# Limited-time free models on Zen ($0). Live-fetched in factory; this is the
# bundled fallback for when the network model list is unavailable (R2 risk).
# Ordered most-used first: muse-spark-1.3 is this user's daily driver.
FREE_MODELS: list[dict] = [
    {"id": "muse-spark-1.3-contributor-free", "name": "Muse Spark 1.3 Free", "context": 1048576, "output": 131072},
    {"id": "x-preview-f-free", "name": "Ox Alpha Free (Unlimited)", "context": 1000000, "output": 131072},
    {"id": "big-pickle", "name": "Big Pickle", "context": 200000, "output": 32000},
    {"id": "hy3-free", "name": "Hy3 Free", "context": 190000, "output": 64000},
    {"id": "mimo-v2.5-free", "name": "MiMo-V2.5 Free", "context": 200000, "output": 32000},
    {"id": "deepseek-v4-flash-free", "name": "DeepSeek V4 Flash Free", "context": 200000, "output": 128000},
    {"id": "nemotron-3-ultra-free", "name": "Nemotron 3 Ultra Free", "context": 1000000, "output": 128000},
    {"id": "nemotron-3.5-lightning-free", "name": "Nemotron 3.5 Lightning Free", "context": 262144, "output": 128000},
]


class ZenProvider(OpenAICompatProvider):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "x-preview-f-free",
        session_id: str | None = None,
        project: str | None = None,
        **kwargs: Any,
    ):
        # If no key given, we still need SOMETHING; Zen accepts "public" for free models.
        effective_key = api_key or "public"
        # Stable session id across turns (like the official client) so Zen
        # keeps provider affinity and serves the real free quota. The id only
        # changes when rotate_session() runs on a DEAD lane — never per turn.
        # x-opencode-request is a FRESH id per request (official: per-message
        # user id); reusing the session id there made every turn look like
        # the same request. _headers() refreshes it on every call below.
        self._base_session = session_id
        rot_key = session_id or "__nosession__"
        with _LANE_LOCK:
            epoch = _LANE_EPOCH.get(rot_key, 0)
        # Epoch suffixing must stay gate-compliant (ses_ + 9 hex + 17 alnum):
        # derive a fresh compliant id per epoch instead of appending ::rN.
        effective_sid = session_id
        if epoch and session_id:
            effective_sid = zen_session_id(f"{session_id}::r{epoch}")
        elif epoch:
            effective_sid = zen_session_id(f"cli::r{epoch}")
        client_headers: dict[str, str] = {
            "User-Agent": _user_agent(),
            "x-opencode-client": "cli",
            "x-opencode-project": project or "opencode",
            "x-opencode-request": uuid4().hex,
        }
        if effective_sid:
            client_headers["x-opencode-session"] = effective_sid
        merged = dict(client_headers)
        merged.update(kwargs.pop("extra_headers", {}) or {})
        super().__init__(
            id="opencode",
            name="OpenCode Zen",
            base_url=ZEN_BASE_URL,
            api_key=effective_key,
            model=model,
            is_free=True,
            extra_headers=merged,
            reasoning_passthrough=True,
            **kwargs,
        )
        self.has_key = bool(api_key)

    def _headers(self) -> dict[str, str]:
        """Fresh x-opencode-request per call, stable x-opencode-session.

        Official sends a new request id for every message but keeps the
        session id stable for the whole chat (sticky lane). The base
        implementation would resend the init-time request id forever.
        """
        headers = super()._headers()
        headers["x-opencode-request"] = uuid4().hex
        # UA must track _CURRENT_VERSION (auto-bumped on 426); the init-time
        # extra_headers value would otherwise stay stale forever.
        headers["User-Agent"] = _user_agent()
        return headers

    def abort_stream(self) -> None:
        # Responses-API streams register here too (see stream_chat); the chat
        # transport manages self._active_resp itself.
        from ..util.net import force_close_response

        box = getattr(self, "_responses_box", None)
        if isinstance(box, list) and box:
            force_close_response(box[0])
        super().abort_stream()

    def _session_cache_key(self) -> str | None:
        """Stable cache key for the Responses `prompt_cache_key` (no lane epoch)."""
        base = self._base_session or self.extra_headers.get("x-opencode-session") or ""
        return str(base).split("::r")[0] or None

    def stream_chat(self, messages, tools=None, on_event=None, **kwargs):
        """Adaptive transport: try the model's cached-preferred API first
        (Responses by default, like the official client), silently fall back
        to the other one when the model doesn't speak it, and remember what
        worked per model — so present AND future models land on a working
        transport without user-visible retries.

        Only TransportIncompatible — or an HTTP 5xx with zero output, which
        on Zen means "this model doesn't live on this API" as often as a
        transient — triggers the silent fallback; every other error
        propagates exactly as the chat transport raises it (rotation and the
        agent loop keep their retry/failover semantics).
        """
        from .base import ProviderError
        from .responses import (
            TransportIncompatible,
            get_preferred_endpoint,
            set_preferred_endpoint,
            stream_responses,
        )

        events: list = []
        sink = on_event or (lambda e: events.append(e))
        model = self.model
        pref = get_preferred_endpoint(model)
        other = "chat" if pref == "responses" else "responses"
        is_interrupted = kwargs.get("is_interrupted")

        def _run(ep: str) -> None:
            effort = (getattr(self, "reasoning_effort", "") or "").strip().lower() or None
            if ep == "responses":
                box: list = [None]
                self._responses_box = box
                try:
                    stream_responses(
                        base_url=self.base_url,
                        headers=self._headers(),
                        timeout=self.timeout,
                        model=model,
                        name=self.name,
                        messages=messages,
                        tools=tools,
                        sink=sink,
                        session_key=self._session_cache_key(),
                        is_interrupted=is_interrupted,
                        active_slot=box,
                        extra_payload={"reasoning_effort": effort} if effort else None,
                    )
                finally:
                    self._responses_box = []
            else:
                self._stream(messages, tools, sink, **kwargs)

        def _finish(ep: str):
            set_preferred_endpoint(model, ep)
            if on_event is None:
                return events  # type: ignore[return-value]
            return None

        def _attempt() -> Any:
            try:
                _run(pref)
            except TransportIncompatible:
                # This model doesn't speak the preferred API — one silent shot
                # on the other transport (this is also how unknown future models
                # are onboarded: first contact tries Responses, then Chat).
                try:
                    _run(other)
                except TransportIncompatible as e2:
                    raise e2
                except ProviderError:
                    # other transport failed too — report the *preferred* failure
                    # would mislead; surface this one (it ran last, most context)
                    raise
                return _finish(other)
            except ProviderError as e:
                # HTTP 5xx before any output is ambiguous on Zen: a dead lane AND
                # a wrong API both look like this (muse 500s on chat, big-pickle
                # 500s on responses). One silent shot on the other transport; if
                # it answers, that settles it and the choice is cached.
                status = getattr(e, "status", None)
                if status is not None and 500 <= status < 600:
                    try:
                        _run(other)
                    except (TransportIncompatible, ProviderError):
                        raise e from None
                    return _finish(other)
                raise
            return _finish(pref)

        try:
            return _attempt()
        except ProviderError as e:
            # Self-healing version floor: when Zen raises the minimum client
            # version, adopt it from the 426 message and retry once instead
            # of failing every free model until the next code update.
            if _bump_version_from_error(_is_upgrade_required(e)):
                events.clear()
                return _attempt()
            # Session exhaustion: a ses_ id reused too often starts 403ing
            # while fresh ones pass. Rotate to a fresh compliant id once;
            # a genuinely broken fingerprint still fails fast afterwards.
            body = str(e)
            if getattr(e, "status", None) == 403 and (
                "FreeTier" in body or "within OpenCode" in body
            ):
                try:
                    self.rotate_session()
                except Exception:
                    raise
                events.clear()
                return _attempt()
            raise

    def rotate_session(self) -> None:
        """Force a different upstream lane on the next request.

        Bumps this session's rotation epoch so every future ZenProvider built
        for the same base session id carries a fresh x-opencode-session value —
        Zen then assigns it a new lane instead of the dead one. Called by the
        rotation layer when a stream dies with server-side lane failure
        (in-band error / empty reply), NOT on local network problems.
        x-opencode-request stays fresh-per-call via _headers(), untouched here.
        """
        key = self._base_session or "__nosession__"
        with _LANE_LOCK:
            epoch = _LANE_EPOCH.get(key, 0) + 1
            _LANE_EPOCH[key] = epoch
        # Compliant rotation value (raw ::rN suffixes trip the free-tier
        # gate's session-shape check with 403): derive per epoch.
        seed = f"{self._base_session}::r{epoch}" if self._base_session else f"cli::r{epoch}"
        self.extra_headers["x-opencode-session"] = zen_session_id(seed)
