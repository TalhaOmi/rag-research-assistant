from types import SimpleNamespace

import pytest

from paper_assistant.ingest import Chunk
from paper_assistant.retrieve import Hit


def make_hit(paper_id: str, page: int, text: str = "some text", rank: int = 1) -> Hit:
    return Hit(Chunk(f"{paper_id}:p{page}:c0", paper_id, page, text), 1.0, rank)


class FakeClient:
    """Stands in for anthropic.Anthropic: returns a canned forced-tool-call payload."""

    def __init__(self, payload: dict):
        self.payload = payload
        self.calls: list[dict] = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        block = SimpleNamespace(
            type="tool_use", name=kwargs["tools"][0]["name"], input=self.payload
        )
        return SimpleNamespace(content=[block])


@pytest.fixture
def hits():
    return [
        make_hit("2304.01543v1", 4, "SHAP values explain model predictions.", 1),
        make_hit("1812.04608v2", 7, "Metrics for explanations include fidelity.", 2),
    ]
