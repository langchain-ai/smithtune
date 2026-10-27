"""Small single-GPU inference server for functional pilots, not throughput serving.

The engine uses the same native renderer/parser as training and the reload smoke.
SageMaker's /ping and /invocations transport is separate from model computation.
No tool calls are executed. Capacity is released through the cloud adapter.
"""
from __future__ import annotations

from contextlib import nullcontext
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import threading

from smithtune.providers.base import PipelineError
from smithtune.providers.baseten.sampling_formats import parse_completion, stop_sequences
from smithtune.rendering import load_training_renderer
from smithtune.training.contracts import verify_artifact
from smithtune.training.inference import generate_tokens, verify_loaded_adapter
from smithtune.training.hf_sft import report_stage

MAX_BODY_BYTES = 6 * 1024 * 1024


class InferenceEngine:
    def __init__(self, model, renderer, spec, *, source_run_id, device, max_context_tokens, max_new_tokens=256):
        if not 2 <= max_context_tokens <= spec.training_context_limit or not 1 <= max_new_tokens <= 1024:
            raise PipelineError("invalid endpoint context/output limits")
        self.model, self.renderer, self.spec = model, renderer, spec
        self.source_run_id, self.device = source_run_id, device
        self.max_context_tokens, self.max_new_tokens = max_context_tokens, max_new_tokens
        stops = renderer.tokenizer.encode(stop_sequences(renderer, spec)[0], add_special_tokens=False)
        if len(stops) != 1:
            raise PipelineError("the initial serving worker requires one native stop token")
        self.eos_token_id = stops[0]

    @classmethod
    def load(cls, directory, *, device="cuda:0", max_context_tokens, max_new_tokens=256):
        report_stage("serving_runtime")
        import torch
        from peft import PeftModel
        from transformers import AutoModelForImageTextToText

        report_stage("serving_verify_artifact")
        result, spec = verify_artifact(directory)
        if device != "cpu" and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
            raise PipelineError("serving requires a CUDA GPU supporting bfloat16")
        report_stage("serving_load_renderer")
        renderer = load_training_renderer(spec)
        report_stage("serving_load_base_model")
        base = AutoModelForImageTextToText.from_pretrained(
            spec.base_model, revision=result["base_model_revision"], trust_remote_code=False,
            dtype=torch.float32 if device == "cpu" else torch.bfloat16, attn_implementation="sdpa",
        ).to(device)
        report_stage("serving_load_adapter")
        model = PeftModel.from_pretrained(base, directory / "adapter", is_trainable=False).eval()
        verify_loaded_adapter(model, directory / "adapter")
        report_stage("serving_model_loaded")
        return cls(model, renderer, spec, source_run_id=result["run_id"], device=device,
                   max_context_tokens=max_context_tokens, max_new_tokens=max_new_tokens)

    def predict(self, payload):
        if not isinstance(payload, dict) or set(payload) - {"messages", "tools", "max_new_tokens", "variant"}:
            raise PipelineError("expected messages, optional tools, max_new_tokens and variant")
        messages, tools = payload.get("messages"), payload.get("tools", [])
        if (not isinstance(messages, list) or not messages or not all(isinstance(row, dict) for row in messages)
                or not isinstance(tools, list) or not all(isinstance(tool, dict) for tool in tools)):
            raise PipelineError("messages and tools must be lists of objects")
        limit, variant = payload.get("max_new_tokens", self.max_new_tokens), payload.get("variant", "tuned")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= self.max_new_tokens:
            raise PipelineError("max_new_tokens exceeds the endpoint's configured limit")
        if variant not in {"base", "tuned"}:
            raise PipelineError("variant must be base or tuned")
        tokens = self.renderer.prompt_tokens(messages, tools=tools)
        if not tokens or len(tokens) + limit > self.max_context_tokens:
            raise PipelineError("prompt plus requested output exceeds endpoint context; no truncation is allowed")
        with self.model.disable_adapter() if variant == "base" else nullcontext():
            generated = generate_tokens(self.model, tokens, device=self.device,
                                        eos_token_id=self.eos_token_id, max_new_tokens=limit)
        candidate = parse_completion(self.renderer, self.spec, generated.pop("tokens"), generated["stop_reason"], tools=tools)
        return {"schema_version": 1, "source_run_id": self.source_run_id, "variant": variant,
                "model": self.spec.base_model, "message": candidate,
                "usage": {"prompt_tokens": generated["prompt_tokens"], "completion_tokens": generated["output_tokens"]},
                "finish_reason": generated["stop_reason"]}


def handler_class(engine):
    gpu_lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(60)

        def log_message(self, *_args):
            # Requests and error messages may contain private trajectories.
            pass

        def respond(self, status, value):
            data = json.dumps(value, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self.respond(200, {"status": "ready"}) if self.path == "/ping" else self.respond(404, {"error": "not_found"})

        def do_POST(self):
            if self.path != "/invocations":
                return self.respond(404, {"error": "not_found"})
            if self.headers.get_content_type() != "application/json":
                return self.respond(415, {"error": "expected_application_json"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                return self.respond(400, {"error": "invalid_content_length"})
            if not 0 < length <= MAX_BODY_BYTES:
                return self.respond(413, {"error": "invalid_body_size"})
            if not gpu_lock.acquire(blocking=False):
                return self.respond(503, {"error": "busy", "retryable": True})
            try:
                payload = json.loads(self.rfile.read(length))
                self.respond(200, engine.predict(payload))
            except (PipelineError, ValueError, TypeError, KeyError):
                self.respond(400, {"error": "invalid_request_or_context", "retryable": False})
            except Exception:
                self.respond(500, {"error": "generation_failed", "retryable": False})
            finally:
                gpu_lock.release()

    return Handler


def serve(directory: Path):
    # Explicit deployment limits; model registry limits are not memory promises.
    engine = InferenceEngine.load(directory, max_context_tokens=int(os.environ["SMITHTUNE_MAX_CONTEXT_TOKENS"]),
                                  max_new_tokens=int(os.environ.get("SMITHTUNE_MAX_NEW_TOKENS", "256")))
    server = ThreadingHTTPServer(("0.0.0.0", 8080), handler_class(engine))
    report_stage("serving_ready")
    try:
        server.serve_forever()
    finally:
        server.server_close()
