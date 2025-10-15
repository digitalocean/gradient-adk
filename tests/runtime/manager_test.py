import asyncio
import types
import importlib
import builtins
import pytest

from gradient_adk.runtime import manager as mgr


class DummyInstrumentor:
    def __init__(self):
        self.installed = 0
        self.uninstalled = 0
        self.attached = []
        self.tracker_seen = None
        self.raise_on_attach = False

    def install(self, tracker):
        self.installed += 1
        self.tracker_seen = tracker

    def uninstall(self):
        self.uninstalled += 1

    def attach_to_graph(self, graph_like):
        if self.raise_on_attach:
            raise RuntimeError("boom")
        self.attached.append(graph_like)


class DummyTracker:
    def __init__(self):
        self.cleared = 0

    def clear_executions(self):
        self.cleared += 1


class DummyDOTracker:
    """Acts as DigitalOceanTracesTracker for isinstance checks."""

    def __init__(self, *a, **kw):
        self.ended = []

    # Signature used by RuntimeManager.end_request
    def end_node_execution(self, node_exec, outputs):
        self.ended.append((node_exec, outputs))


@pytest.fixture(autouse=True)
def _isolate_globals(monkeypatch):
    """
    Keep globals clean across tests and avoid real atexit registration.
    """
    # Avoid registering real atexit handlers in tests
    monkeypatch.setattr(mgr.atexit, "register", lambda *_a, **_kw: None)

    # Reset global runtime manager per test
    monkeypatch.setattr(mgr, "_runtime_manager", None, raising=True)

    # Replace default instrumentor with a controllable dummy
    monkeypatch.setattr(mgr, "LangGraphInstrumentor", DummyInstrumentor, raising=True)


@pytest.fixture
def ctx_spy(monkeypatch):
    """
    Spy on start/end request context calls.
    """
    calls = {"start": [], "end": []}

    def _start(name, inputs, metadata):
        calls["start"].append({"name": name, "inputs": inputs, "metadata": metadata})
        return object()  # non-None (truthy) context

    def _end(outputs, error):
        calls["end"].append({"outputs": outputs, "error": error})
        return object()  # non-None (truthy) context

    monkeypatch.setattr(mgr, "start_request_context", _start, raising=True)
    monkeypatch.setattr(mgr, "end_request_context", _end, raising=True)
    return calls


def test_install_uninstall_idempotent():
    tr = DummyTracker()
    rm = mgr.RuntimeManager(tracker=tr)

    # One dummy instrumentor created by _register_default_instrumentors
    assert len(rm._instrumentors) == 1
    inst: DummyInstrumentor = rm._instrumentors[0]
    assert inst.installed == 0
    assert inst.uninstalled == 0

    rm.install_instrumentation()
    rm.install_instrumentation()  # idempotent

    assert rm._installed is True
    assert inst.installed == 1  # only once
    assert inst.tracker_seen is tr

    rm.uninstall_instrumentation()
    rm.uninstall_instrumentation()  # idempotent

    assert rm._installed is False
    assert inst.uninstalled == 1  # only once


def test_start_request_installs_and_clears_and_sets_context(ctx_spy):
    tr = DummyTracker()
    rm = mgr.RuntimeManager(tracker=tr)

    # Not installed yet
    assert rm._installed is False

    rm.start_request("MyEntrypoint", inputs={"x": 1}, metadata={"m": 2})

    # Installed via start_request
    assert rm._installed is True
    # Tracker cleared
    assert tr.cleared == 1
    # Context started as expected
    assert ctx_spy["start"] == [
        {"name": "MyEntrypoint", "inputs": {"x": 1}, "metadata": {"m": 2}}
    ]


def test_end_request_default_tracker_does_not_trigger_do(mocker, ctx_spy, monkeypatch):
    tr = DummyTracker()
    rm = mgr.RuntimeManager(tracker=tr)

    # Ensure isinstance(...) check won't match DummyTracker
    class MarkerDO(mgr.DigitalOceanTracesTracker):
        pass

    # Replace DigitalOceanTracesTracker with a unique class to be safe
    monkeypatch.setattr(mgr, "DigitalOceanTracesTracker", MarkerDO, raising=True)

    # Spy to ensure no end_node_execution is attempted on tracker
    spy = mocker.spy(tr, "clear_executions")  # just a harmless spy

    rm.end_request(outputs={"ok": True})
    # Context ended
    assert ctx_spy["end"] == [{"outputs": {"ok": True}, "error": None}]
    # No attribute end_node_execution on DummyTracker => would raise if called


def test_end_request_with_digitalocean_tracker_triggers_submit(ctx_spy, monkeypatch):
    # Patch the class in the module that end_request() imports at runtime
    import importlib

    do_mod = importlib.import_module("gradient_adk.runtime.digitalocean_tracker")
    monkeypatch.setattr(
        do_mod, "DigitalOceanTracesTracker", DummyDOTracker, raising=True
    )

    do_tr = DummyDOTracker()
    rm = mgr.RuntimeManager(tracker=do_tr)

    rm.end_request(outputs={"final": 42})

    # Context ended
    assert ctx_spy["end"] == [{"outputs": {"final": 42}, "error": None}]
    # Submission triggered exactly once with our outputs
    assert len(do_tr.ended) == 1
    node_exec, outputs = do_tr.ended[0]
    assert outputs == {"final": 42}
    assert node_exec is not None


def test_attach_to_graph_calls_attach_and_swallows_errors():
    tr = DummyTracker()
    rm = mgr.RuntimeManager(tracker=tr)

    # Replace default list with two instrumentors: one OK, one failing
    ok_inst = DummyInstrumentor()
    bad_inst = DummyInstrumentor()
    bad_inst.raise_on_attach = True
    rm._instrumentors = [ok_inst, bad_inst]

    g = object()
    rm.attach_to_graph(g)

    # attach_to_graph auto-installs
    assert rm._installed is True

    # OK instrumentor attached
    assert ok_inst.attached == [g]
    # Bad instrumentor exception is swallowed; test passes if no exception raised


def test_attach_graph_wrapper_calls_manager(monkeypatch):
    captured = {"graph": None}

    class FakeRM:
        def attach_to_graph(self, graph_like):
            captured["graph"] = graph_like

    monkeypatch.setattr(mgr, "get_runtime_manager", lambda: FakeRM(), raising=True)

    g = {"hello": "world"}
    mgr.attach_graph(g)

    assert captured["graph"] is g


def test_get_runtime_manager_singleton():
    a = mgr.get_runtime_manager()
    b = mgr.get_runtime_manager()
    assert isinstance(a, mgr.RuntimeManager)
    assert a is b  # singleton


def test_configure_digitalocean_traces_sets_tracker(monkeypatch):
    # Patch DO client and tracker to dummies
    class DummyClient:
        def __init__(self, api_token):
            self.api_token = api_token

    created = {}

    def dummy_tracker_factory(
        client, agent_workspace_name, agent_deployment_name, enable_auto_submit
    ):
        # Verify parameters are threaded through
        created["client"] = client
        created["ws"] = agent_workspace_name
        created["dep"] = agent_deployment_name
        created["auto"] = enable_auto_submit
        return DummyDOTracker()

    monkeypatch.setattr(mgr, "AsyncDigitalOceanGenAI", DummyClient, raising=True)
    monkeypatch.setattr(
        mgr,
        "DigitalOceanTracesTracker",
        lambda **kw: dummy_tracker_factory(**kw),
        raising=True,
    )

    rm = mgr.configure_digitalocean_traces(
        api_token="tok",
        agent_workspace_name="ws",
        agent_deployment_name="dep",
        enable_auto_submit=True,
    )

    assert isinstance(rm, mgr.RuntimeManager)
    # ensure global was set
    assert mgr.get_runtime_manager() is rm
    # verify DO wiring
    assert isinstance(rm.get_tracker(), DummyDOTracker)
    assert isinstance(created["client"], DummyClient)
    assert created["client"].api_token == "tok"
    assert created["ws"] == "ws"
    assert created["dep"] == "dep"
    assert created["auto"] is True


@pytest.mark.asyncio
async def test_run_entrypoint_async_success(ctx_spy):
    tr = DummyTracker()
    rm = mgr.RuntimeManager(tracker=tr)

    @pytest.mark.asyncio
    async def ep(data, context):
        return {"res": data["x"] + 1}

    out = await rm.run_entrypoint(ep, {"x": 2}, context={"y": 3})
    assert out == {"res": 3}

    # start + end called once each
    assert len(ctx_spy["start"]) == 1
    assert len(ctx_spy["end"]) == 1
    assert ctx_spy["end"][0]["outputs"] == {"res": 3}
    assert ctx_spy["end"][0]["error"] is None


@pytest.mark.asyncio
async def test_run_entrypoint_sync_error(ctx_spy):
    tr = DummyTracker()
    rm = mgr.RuntimeManager(tracker=tr)

    def ep(data, context):
        raise ValueError("nope")

    with pytest.raises(ValueError, match="nope"):
        await rm.run_entrypoint(ep, {"x": 1}, context=None)

    # end_request called with error populated
    assert len(ctx_spy["end"]) == 1
    assert ctx_spy["end"][0]["outputs"] is None
    assert "nope" in ctx_spy["end"][0]["error"]
