"""Tests for traces service browser opening."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from gradient_adk.cli.agent import traces_service


class ImmediateTimer:
    """Timer stub that runs the cleanup immediately in tests."""

    def __init__(self, delay, func, args=None, kwargs=None):
        self.func = func
        self.args = args or ()
        self.kwargs = kwargs or {}

    def start(self):
        self.func(*self.args, **self.kwargs)


def test_linux_prefers_temp_file(monkeypatch):
    """Linux should skip data: URLs and open a file:// temp page instead."""
    opened_urls = []

    def fake_open(url, new=0, autoraise=True):
        opened_urls.append(url)
        return True

    monkeypatch.setattr(traces_service.sys, "platform", "linux")
    monkeypatch.setattr(traces_service.webbrowser, "open", fake_open)
    monkeypatch.setattr(traces_service.threading, "Timer", ImmediateTimer)

    service = traces_service.GalileoTracesService(client=MagicMock())
    service._open_traces_console(
        base_url="https://example.com",
        access_token="token",
        logstream_id="log-id",
        project_id="proj-id",
        workspace_name="workspace",
        agent_name="agent",
    )

    assert len(opened_urls) == 1
    assert opened_urls[0].startswith("file://")


def test_open_failure_raises_and_cleans_temp(monkeypatch):
    """If the browser cannot be opened, the service should raise and clean temp files."""
    created_files = []
    real_named_tempfile = traces_service.tempfile.NamedTemporaryFile

    def fake_named_tempfile(*args, **kwargs):
        f = real_named_tempfile(*args, **kwargs)
        created_files.append(f.name)
        return f

    monkeypatch.setattr(traces_service.sys, "platform", "linux")
    monkeypatch.setattr(traces_service.tempfile, "NamedTemporaryFile", fake_named_tempfile)
    monkeypatch.setattr(traces_service.webbrowser, "open", lambda *_, **__: False)
    monkeypatch.setattr(traces_service.threading, "Timer", ImmediateTimer)

    service = traces_service.GalileoTracesService(client=MagicMock())

    with pytest.raises(RuntimeError):
        service._open_traces_console(
            base_url="https://example.com",
            access_token="token",
            logstream_id="log-id",
            project_id="proj-id",
            workspace_name="workspace",
            agent_name="agent",
        )

    for path in created_files:
        assert not Path(path).exists()
