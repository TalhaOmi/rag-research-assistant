import pytest

pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from paper_assistant.api import app, get_assistant  # noqa: E402
from paper_assistant.generate import Answer, Citation  # noqa: E402

from .conftest import make_hit  # noqa: E402


class StubAssistant:
    titles = {"2304.01543v1": "A Brief Review"}

    def __init__(self, answer: Answer | Exception):
        self.answer = answer

    def ask(self, question, k=5, mode="hybrid"):
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


@pytest.fixture
def client():
    yield TestClient(app)
    app.dependency_overrides.clear()


def use(answer):
    app.dependency_overrides[get_assistant] = lambda: StubAssistant(answer)


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_ask_returns_answer_with_citations_and_sources(client):
    hit = make_hit("2304.01543v1", 4, "SHAP text")
    use(Answer("q", "SHAP [2304.01543v1, p.4].", True, [Citation("2304.01543v1", 4)], [], [hit]))
    body = client.post("/ask", json={"question": "What is SHAP?"}).json()
    assert body["answerable"] is True
    assert body["citations"][0]["page"] == 4
    assert body["citations"][0]["title"] == "A Brief Review"
    assert body["citations"][0]["url"].endswith("2304.01543v1#page=4")
    assert body["sources"][0]["text"] == "SHAP text"


def test_refusal_is_returned_not_an_error(client):
    use(Answer("q", "Not in the sources.", False, [], [], []))
    response = client.post("/ask", json={"question": "Who won the World Cup?"})
    assert response.status_code == 200
    assert response.json()["answerable"] is False


@pytest.mark.parametrize("payload", [{"question": ""}, {"question": "ok?", "k": 0}, {}])
def test_invalid_requests_get_422(client, payload):
    use(Answer("q", "x", False, [], [], []))
    assert client.post("/ask", json=payload).status_code == 422


def test_missing_index_is_503(client):
    use(FileNotFoundError("no index"))
    assert client.post("/ask", json={"question": "anything"}).status_code == 503


def test_generation_failure_is_502(client):
    use(RuntimeError("boom"))
    assert client.post("/ask", json={"question": "anything"}).status_code == 502
