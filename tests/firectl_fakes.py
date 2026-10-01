"""A firectl stand-in shaped like firectl 1.8.9 output (match, deployment get)."""
import json
import subprocess
from types import SimpleNamespace


def shape(display_name, accelerator, *, validated=True, latest=True, name=None):
    name = name or f"accounts/fireworks/deploymentShapes/{display_name.lower().replace(' ', '-')}"
    return {"name": f"{name}/versions/v1", "validated": validated, "latest_validated": latest, "public": True,
            "snapshot": {"name": name, "display_name": display_name, "accelerator_count": 1,
                         "accelerator_type": accelerator, "base_model": "accounts/fireworks/models/base"}}


# Verbatim from firectl 1.8.9 when a mutating command runs under an AI agent.
AGENT_BLOCK = ('Failed to execute: BLOCKED: mutating command "firectl deployment {verb}" cannot run inside an AI agent '
               'for account "account-id".\nRun this command manually in your terminal:\n')


class FakeFirectl:
    def __init__(self, *, shapes=None, ready_after=0, state="READY", events=None, block_agents=False, fail=None):
        self.shapes = shapes if shapes is not None else [shape("Base 1x H100", "NVIDIA_H100_80GB")]
        self.ready_after, self.state, self.gets = ready_after, state, 0
        self.commands = []
        self.events = events
        self.block_agents, self.fail = block_agents, fail

    def __call__(self, command, capture=False, **kwargs):
        self.commands.append(command)
        if command[1:2] == ["deployment"] and command[2] in {"create", "delete"}:
            if self.block_agents:
                raise subprocess.CalledProcessError(1, command, output="", stderr=AGENT_BLOCK.format(verb=command[2]))
            if self.fail:
                raise subprocess.CalledProcessError(1, command, output="", stderr=self.fail)
        if command[1:3] == ["deployment-shape-version", "match"]:
            return SimpleNamespace(stdout=json.dumps(self.shapes), stderr="")
        if command[1:3] == ["deployment", "get"]:
            self.gets += 1
            ready = {"ready_replica_count": 1} if self.gets > self.ready_after else {}
            return SimpleNamespace(stdout=json.dumps({"state": self.state, "replica_stats": ready}), stderr="")
        if command[1:3] == ["deployment", "create"] and self.events is not None:
            self.events.append(("deploy", command))
        return SimpleNamespace(stdout="", stderr="")
