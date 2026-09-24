"""Tests for the Zen free-tier gate hardening: version parsing/auto-bump,
session-id shape, and the --check live probe classification (all mocked)."""

import re

import pytest

from opencode_py.providers import base as B
from opencode_py.providers import zen as Z


def test_parse_version():
    assert Z._parse_version("OpenCode 1.18.0 or newer is required") == (1, 18, 0)
    assert Z._parse_version("opencode/1.18.32") == (1, 18, 32)
    assert Z._parse_version("no version here") is None


def test_bump_version_from_error(monkeypatch):
    monkeypatch.setattr(Z, "_CURRENT_VERSION", "1.18.32")
    assert Z._bump_version_from_error("OpenCode 1.19.0 or newer is required") is True
    assert Z._user_agent() == "opencode/1.19.0"
    # lower floor: no bump
    assert Z._bump_version_from_error("OpenCode 1.18.0 or newer is required") is False
    assert Z._user_agent() == "opencode/1.19.0"
    # garbage: no bump
    assert Z._bump_version_from_error("Internal server error") is False


def test_426_auto_retry_bumps_and_resucceeds(monkeypatch):
    monkeypatch.setattr(Z, "_CURRENT_VERSION", "1.18.32")
    p = Z.ZenProvider(api_key=None, model="m", session_id=None)
    calls = []

    def fake_stream(messages, tools, sink, **kwargs):
        calls.append(dict(sink=sink))
        if len(calls) == 1:
            raise B.ProviderError("OpenCode 1.99.0 or newer is required", status=426)
        from opencode_py.providers.base import ProviderEvent

        sink(ProviderEvent(kind="text_delta", text="hi"))

    monkeypatch.setattr(p, "_stream", fake_stream)
    monkeypatch.setattr(
        "opencode_py.providers.responses.get_preferred_endpoint", lambda m: "chat"
    )
    out = p.stream_chat([{"role": "user", "content": "hi"}])
    assert Z._user_agent() == "opencode/1.99.0"
    assert len(calls) == 2
    assert out and out[0].text == "hi"


def test_zen_session_id_shape_and_stability():
    pat = re.compile(r"^ses_[0-9a-f]{9}[0-9A-Za-z]{17}$")
    a = Z.zen_session_id("48680f5de4fb44a4b9b97b2c9a2a8875")
    assert pat.match(a)
    assert Z.zen_session_id("48680f5de4fb44a4b9b97b2c9a2a8875") == a
    assert pat.match(Z.zen_session_id(None))
    assert pat.match(Z.zen_session_id("health-mimo-x"))
    good = "ses_f31b42f72ffekqYrVCUYnT06PI"
    assert Z.zen_session_id(good) == good


def _probe_with(monkeypatch, exc):
    from opencode_py.config import Config
    from opencode_py.providers import rotation as R

    class FakeProvider:
        def stream_chat(self, messages, tools, on_event, **kwargs):
            raise exc

    monkeypatch.setattr(R, "build_provider", lambda *a, **k: FakeProvider())
    cfg = Config()
    cfg.provider = "opencode"
    cfg.model = "opencode/big-pickle"
    return R.probe_zen(cfg, auth=None)


def test_probe_classifies_rate_limited(monkeypatch):
    out = _probe_with(monkeypatch, B.RateLimitError("429 FreeUsageLimitError"))
    assert out["status"] == "rate_limited"


def test_probe_classifies_upgrade(monkeypatch):
    out = _probe_with(
        monkeypatch, B.ProviderError("OpenCode 1.99.0 or newer is required", status=426)
    )
    assert out["status"] == "upgrade"
    assert "1.99.0" in out["detail"]


def test_probe_classifies_free_tier(monkeypatch):
    out = _probe_with(
        monkeypatch,
        B.ProviderError("OpenCode's free tier can only be used from within OpenCode", status=403),
    )
    assert out["status"] == "free_tier"


def test_probe_classifies_transient(monkeypatch):
    out = _probe_with(
        monkeypatch, B.ProviderError("Internal server error", status=500)
    )
    assert out["status"] == "transient"


def test_probe_ok_on_text(monkeypatch):
    from opencode_py.config import Config
    from opencode_py.providers import rotation as R

    class FakeProvider:
        def stream_chat(self, messages, tools, on_event, **kwargs):
            on_event(B.ProviderEvent(kind="text_delta", text="hi"))

    monkeypatch.setattr(R, "build_provider", lambda *a, **k: FakeProvider())
    cfg = Config()
    cfg.provider = "opencode"
    cfg.model = "opencode/big-pickle"
    out = R.probe_zen(cfg, auth=None)
    assert out["status"] == "ok"


def test_gate_hint_per_cause():
    assert "OFFICIAL_VERSION" in (Z.gate_hint(426, "OpenCode 9.9.9 or newer") or "")
    assert "--check" in (Z.gate_hint(403, "FreeTierError within OpenCode") or "")
    assert "quota" in (Z.gate_hint(429, "rate limited") or "").lower()
    assert "wait and retry" in (Z.gate_hint(500, "server error") or "")
    assert Z.gate_hint(400, "bad request") is None


def test_403_rotates_session_once_then_succeeds(monkeypatch):
    p = Z.ZenProvider(api_key=None, model="m", session_id=None)
    before = p.extra_headers.get("x-opencode-session")
    calls = []

    def fake_stream(messages, tools, sink, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise B.ProviderError("FreeTierError within OpenCode", status=403)
        from opencode_py.providers.base import ProviderEvent

        sink(ProviderEvent(kind="text_delta", text="ok"))

    monkeypatch.setattr(p, "_stream", fake_stream)
    monkeypatch.setattr(
        "opencode_py.providers.responses.get_preferred_endpoint", lambda m: "chat"
    )
    out = p.stream_chat([{"role": "user", "content": "hi"}])
    assert len(calls) == 2
    after = p.extra_headers.get("x-opencode-session")
    assert after and after != before
    assert out and out[0].text == "ok"
