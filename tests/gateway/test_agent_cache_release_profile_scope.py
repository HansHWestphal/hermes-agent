"""Multiplex invariant: agent-cache eviction commits end-of-session memory under the OWNING profile.

``_spawn_release_thread`` used to start a bare ``threading.Thread``; threads begin with an EMPTY
context, so provider ``on_session_end`` (credentials/home read at call time) ran under the launch
profile — a secondary's transcript was extracted into the default profile's memory namespace, or
failed closed (``UnscopedSecretError``) and the memories were lost.
"""
from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from types import SimpleNamespace

from agent import secret_scope
from gateway.config import GatewayConfig
from gateway.run import GatewayRunner
from gateway.session_execution import GatewaySessionAgentMixin
from hermes_constants import get_hermes_home
from hermes_state_keys import profile_from_session_key


def _runner(
    profile_homes: dict[str, Path], *, multiplex_profiles: bool = True, routing_home: Path | None = None,
) -> GatewayRunner:
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=multiplex_profiles)
    runner.session_store = SimpleNamespace(
        _profile_home_for_key=lambda key: profile_homes.get(key),
        _profile_from_session_key=profile_from_session_key,
        _routing_home=routing_home,
    )
    return runner


def _seen_after_release(runner, key, *, wait_scope=None) -> dict:
    seen: dict = {}
    done = threading.Event()

    def target(agent, k):
        seen["scope"] = secret_scope.current_secret_scope()
        seen["home"] = get_hermes_home()
        done.set()

    runner._spawn_release_thread(target, (None, key), f"t-{key}", inline_fallback=False, session_key=key)
    assert done.wait(5)
    return seen


def test_release_thread_writes_and_reads_only_the_owning_profile_database(tmp_path, monkeypatch):
    default_home = tmp_path / ".hermes"
    owner_home = default_home / "profiles" / "b"
    owner_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    secret_scope.set_multiplex_active(True)
    done = threading.Event()
    seen: dict = {}
    session_key = "agent:b:telegram:dm:1"

    def write_marker():
        database = get_hermes_home() / "memory-test.sqlite3"
        with sqlite3.connect(database) as connection:
            connection.execute("CREATE TABLE markers (value TEXT NOT NULL)")
            connection.execute("INSERT INTO markers VALUES (?)", ("synthetic-profile-marker",))
        seen["database"] = database
        done.set()

    try:
        runner = _runner({session_key: owner_home})
        runner._spawn_release_thread(
            write_marker, (), "test-profile-db-write", inline_fallback=False, session_key=session_key,
        )
        assert done.wait(5)
    finally:
        secret_scope.set_multiplex_active(False)

    database = owner_home / "memory-test.sqlite3"
    assert seen["database"] == database
    assert not (default_home / "memory-test.sqlite3").exists()
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT value FROM markers").fetchone() == ("synthetic-profile-marker",)


def test_release_thread_does_not_write_when_profile_identity_is_missing_or_mismatched(tmp_path, monkeypatch):
    default_home = tmp_path / ".hermes"
    owner_home = default_home / "profiles" / "b"
    owner_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    secret_scope.set_multiplex_active(True)
    database = default_home / "memory-test.sqlite3"

    def write_marker():
        with sqlite3.connect(get_hermes_home() / "memory-test.sqlite3") as connection:
            connection.execute("CREATE TABLE markers (value TEXT NOT NULL)")
            connection.execute("INSERT INTO markers VALUES (?)", ("must-not-write",))

    try:
        _runner({"agent:b:telegram:dm:1": owner_home})._run_release_in_profile_scope(
            write_marker, (), None,
        )
        _runner({"agent:a:telegram:dm:1": owner_home})._run_release_in_profile_scope(
            write_marker, (), "agent:a:telegram:dm:1",
        )
    finally:
        secret_scope.set_multiplex_active(False)

    assert not database.exists()
    assert not (owner_home / "memory-test.sqlite3").exists()


def test_unscoped_housekeeping_sweep_enters_the_owning_profile_scope(tmp_path, monkeypatch):
    default_home = tmp_path / ".hermes"
    prof_b = default_home / "profiles" / "b"
    prof_b.mkdir(parents=True)
    (prof_b / ".env").write_text("HINDSIGHT_LLM_API_KEY=key-of-b\n")
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    secret_scope.set_multiplex_active(True)
    try:
        # The housekeeping watcher runs with NO scope installed.
        assert secret_scope.current_secret_scope() is None
        seen = _seen_after_release(_runner({"agent:b:telegram:dm:1": prof_b}), "agent:b:telegram:dm:1")
    finally:
        secret_scope.set_multiplex_active(False)
    assert seen["home"] == prof_b
    assert seen["scope"] and seen["scope"].get("HINDSIGHT_LLM_API_KEY") == "key-of-b"


def test_in_turn_cap_eviction_keeps_the_callers_scope(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    secret_scope.set_multiplex_active(True)
    token = secret_scope.set_secret_scope({"MARKER": "turn-scope"}, profile_home=str(tmp_path))
    try:
        seen = _seen_after_release(_runner({}), "agent:main:cli:1")
    finally:
        secret_scope.reset_secret_scope(token)
        secret_scope.set_multiplex_active(False)
    assert seen["scope"] == {"MARKER": "turn-scope"}


def test_in_turn_cap_eviction_of_another_profiles_agent_enters_the_owners_scope(tmp_path, monkeypatch):
    """The LRU cap is enforced inside the REQUESTING turn (profile A's scope), and the agent it
    evicts may be profile B's. B's end-of-session commit must run under B, not under A."""
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    default_home = tmp_path / ".hermes"
    prof_a, prof_b = default_home / "profiles" / "a", default_home / "profiles" / "b"
    prof_a.mkdir(parents=True), prof_b.mkdir(parents=True)
    (prof_b / ".env").write_text("HINDSIGHT_LLM_API_KEY=key-of-b\n")
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    secret_scope.set_multiplex_active(True)
    home_token = set_hermes_home_override(str(prof_a))
    scope_token = secret_scope.set_secret_scope(
        {"HINDSIGHT_LLM_API_KEY": "key-of-a"}, profile_home=str(prof_a))
    try:
        seen = _seen_after_release(_runner({"agent:b:telegram:dm:1": prof_b}), "agent:b:telegram:dm:1")
    finally:
        secret_scope.reset_secret_scope(scope_token)
        reset_hermes_home_override(home_token)
        secret_scope.set_multiplex_active(False)
    assert seen["home"] == prof_b
    assert seen["scope"].get("HINDSIGHT_LLM_API_KEY") == "key-of-b"


def test_in_turn_cap_eviction_of_a_default_profile_agent_leaves_the_secondarys_scope(tmp_path, monkeypatch):
    """Inverse: secondary A's turn evicts a default-profile session (``agent:main:`` — no named
    owner in the key). The commit runs under the default home, not under A."""
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    default_home = tmp_path / ".hermes"
    prof_a = default_home / "profiles" / "a"
    prof_a.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    secret_scope.set_multiplex_active(True)
    home_token = set_hermes_home_override(str(prof_a))
    scope_token = secret_scope.set_secret_scope({"MARKER": "profile-a"}, profile_home=str(prof_a))
    try:
        seen = _seen_after_release(_runner({}), "agent:main:telegram:dm:9")
    finally:
        secret_scope.reset_secret_scope(scope_token)
        reset_hermes_home_override(home_token)
        secret_scope.set_multiplex_active(False)
    assert seen["home"] == default_home
    assert seen["scope"] is not None and seen["scope"].get("MARKER") is None


def test_default_profile_owner_is_the_root_even_when_launched_under_a_named_profile(tmp_path, monkeypatch):
    """``hermes -p x gateway`` sets HERMES_HOME to x's home and serves the default profile as a
    secondary. An ``agent:main:`` session still belongs to the default profile at the ROOT, not to
    the launch profile x — otherwise x's turn would commit default's transcript under x."""
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    root = tmp_path / ".hermes"
    prof_x = root / "profiles" / "x"
    prof_x.mkdir(parents=True)
    (root / ".env").write_text("MARKER=default-root\n")
    monkeypatch.setenv("HERMES_HOME", str(prof_x))  # launched under x
    secret_scope.set_multiplex_active(True)
    home_token = set_hermes_home_override(str(prof_x))
    scope_token = secret_scope.set_secret_scope({"MARKER": "profile-x"}, profile_home=str(prof_x))
    try:
        seen = _seen_after_release(_runner({}), "agent:main:telegram:dm:3")
    finally:
        secret_scope.reset_secret_scope(scope_token)
        reset_hermes_home_override(home_token)
        secret_scope.set_multiplex_active(False)
    assert seen["home"] == root
    assert seen["scope"].get("MARKER") == "default-root"


def test_standalone_gateway_legacy_main_key_uses_captured_launch_profile_home(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    profile_home = root / "profiles" / "hve-cfo"
    profile_home.mkdir(parents=True)
    (profile_home / ".env").write_text("MARKER=hve-cfo-launch-home\n")
    monkeypatch.setenv("HERMES_HOME", str(root))
    seen = _seen_after_release(
        _runner({}, multiplex_profiles=False, routing_home=profile_home),
        "agent:main:local:dm:1",
    )
    assert seen["home"] == profile_home
    assert seen["scope"].get("MARKER") == "hve-cfo-launch-home"


def test_unresolved_multiplexed_owner_fails_closed_without_default_fallback(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    root.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(root))
    secret_scope.set_multiplex_active(True)
    try:
        for key in (None, "not-a-session-key", "agent:missing:local:dm:1"):
            called = []
            _runner({})._run_release_in_profile_scope(lambda: called.append(True), (), key)
            assert not called
    finally:
        secret_scope.set_multiplex_active(False)


def test_multiplexed_profile_key_and_resolved_home_must_match(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    profile_b = root / "profiles" / "b"
    profile_b.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(root))
    secret_scope.set_multiplex_active(True)
    try:
        called = []
        _runner({"agent:a:local:dm:1": profile_b})._run_release_in_profile_scope(
            lambda: called.append(True), (), "agent:a:local:dm:1")
    finally:
        secret_scope.set_multiplex_active(False)
    assert not called


def test_standalone_missing_launch_home_fails_closed(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    root.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(root))
    called = []
    _runner({}, multiplex_profiles=False)._run_release_in_profile_scope(
        lambda: called.append(True), (), "agent:main:local:dm:1")
    assert not called


def test_standalone_session_profile_mismatch_fails_closed(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    profile_home = root / "profiles" / "hve-cfo"
    profile_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(root))
    called = []
    _runner({}, multiplex_profiles=False, routing_home=profile_home)._run_release_in_profile_scope(
        lambda: called.append(True), (), "agent:other:local:dm:1")
    assert not called


def test_evicted_agent_release_forwards_session_identity():
    session_key = "agent:main:local:dm:1"
    calls = []

    class Runner:
        def _release_evicted_agent_soft(self, agent):
            pass

        def _spawn_release_thread(self, target, args, name, *, inline_fallback, session_key=None):
            calls.append((target, args, inline_fallback, session_key))

    session = SimpleNamespace(_runner=Runner(), _ctx=SimpleNamespace(session_key=session_key))
    agent = object()
    GatewaySessionAgentMixin._release_evicted_agent(session, agent)
    assert calls == [(session._runner._release_evicted_agent_soft, (agent,), True, session_key)]
