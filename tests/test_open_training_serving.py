"""Local HTTP contract and portable-worker checks; no cloud calls or downloads."""
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
import json
import runpy
import sys
import threading

import pytest

from smithtune.providers.base import PipelineError
from smithtune.training import hf_sft, serving, worker
from smithtune.training.contracts import WorkerSettings


@pytest.fixture
def local_server():
    entered, release = threading.Event(), threading.Event()
    class Engine:
        def predict(self, value):
            if value.get("block"):
                entered.set()
                if not release.wait(5):
                    raise TimeoutError()
            if value.get("fail"):
                raise RuntimeError("private prompt must not appear in response")
            return {"message": "hello"}
    server = ThreadingHTTPServer(("127.0.0.1", 0), serving.handler_class(Engine()))
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    def request(method, path, body=None, headers=None):
        connection = HTTPConnection(*server.server_address, timeout=5)
        try:
            connection.request(method, path, body=body, headers=headers or {"Content-Type": "application/json"})
            result = connection.getresponse()
            return result.status, json.loads(result.read())
        finally:
            connection.close()
    try:
        yield request, entered, release
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_sagemaker_http_protocol_and_controlled_errors(local_server, capsys):
    request, _, _ = local_server
    assert request("GET", "/ping") == (200, {"status": "ready"})
    assert request("POST", "/invocations", '{}')[0] == 200
    assert request("POST", "/invocations", 'broken')[0] == 400
    assert request("POST", "/invocations", '{}', {"Content-Type": "text/plain"})[0] == 415
    assert request("POST", "/invocations", '{}', {"Content-Type": "application/json", "Content-Length": "9999999"})[0] == 413
    assert request("GET", "/unknown")[0] == 404
    assert request("POST", "/invocations", '{"fail":true}') == (500, {"error": "generation_failed", "retryable": False})
    assert not capsys.readouterr().err


def test_busy_endpoint_preserves_single_gpu_execution(local_server):
    request, entered, release = local_server
    results = []
    active = threading.Thread(target=lambda: results.append(request("POST", "/invocations", '{"block":true}')))
    active.start()
    try:
        assert entered.wait(5)
        assert request("POST", "/invocations", '{}') == (503, {"error": "busy", "retryable": True})
        assert request("GET", "/ping")[0] == 200
    finally:
        release.set()
        active.join(timeout=5)
    assert results[0][0] == 200
    assert request("POST", "/invocations", '{}')[0] == 200


def test_portable_module_entrypoint_resolves_worker_before_calling_it(monkeypatch, tmp_path):
    def reached_bundle(_path):
        raise PipelineError("reached bundle validation")
    monkeypatch.setattr(hf_sft, "_load_json", reached_bundle)
    monkeypatch.setattr(sys, "argv", ["worker", "--input-dir", str(tmp_path), "--model-dir", str(tmp_path / "model"),
                                      "--checkpoint-dir", str(tmp_path / "checkpoints"), "--device", "cpu"])
    with pytest.raises(PipelineError, match="reached bundle validation"):
        runpy.run_path(worker.__file__, run_name="__main__")


@pytest.mark.parametrize("settings", [{"batch_size": True}, {"max_epochs": 0}, {"learning_rate": float("nan")},
                                     {"early_stopping_min_delta": float("inf")}, {"trainer_backend": "unknown"}])
def test_shared_worker_rejects_invalid_settings(settings):
    with pytest.raises(PipelineError):
        WorkerSettings(**settings).validate()
