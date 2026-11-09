from fastapi.testclient import TestClient

from app.main import create_app


def test_health_and_chat_api() -> None:
    with TestClient(create_app()) as client:
        health = client.get("/api/v1/health")
        response = client.post(
            "/api/v1/chat",
            json={"query": "同一PON口下多台ONU同时离线并报PON_LOS"},
        )

    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert response.status_code == 200
    assert response.json()["route"] == "rag"
    assert response.json()["citations"]


def test_websocket_emits_pipeline_and_stream_events() -> None:
    with TestClient(create_app()) as client:
        with client.websocket_connect("/ws/chat/test-session") as websocket:
            websocket.send_json({"query": "LOS是什么意思？"})
            events = []
            while "answer_completed" not in events:
                events.append(websocket.receive_json()["event"])

    assert "intent_completed" in events
    assert "route_completed" in events
    assert "answer_delta" in events
    assert events[-1] == "answer_completed"
