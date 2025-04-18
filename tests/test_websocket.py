import pytest
import json
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient
from demo import app

client = TestClient(app)

class TestWebsocket:
    @pytest.mark.asyncio
    @patch('demo.websockets.connect', new_callable=AsyncMock)
    async def test_media_stream_connection(self, mock_connect):
        mock_ws = AsyncMock()
        mock_connect.return_value.__aenter__.return_value = mock_ws

        with client.websocket_connect("/media-stream/test_call_id") as websocket:
            websocket.send_text(json.dumps({
                "event": "start",
                "start": {"streamSid": "test_stream"}
            }))
            websocket.send_text(json.dumps({
                "event": "media",
                "media": {"payload": "base64audio"}
            }))

            # You can also test expected behavior from the server side here
            # Example:
            # response = websocket.receive_text()
            # assert "something" in response


    @pytest.mark.asyncio
    async def test_call_updates_connection(self):
        with client.websocket_connect("/call-updates/test_call_id") as websocket:
            # Receive the welcome message
            data = websocket.receive_text()
            payload = json.loads(data)
            assert payload["event_type"] == "connected"
            assert payload["message"] == "Connected to call updates"

            # Send a test message
            websocket.send_text(json.dumps({"event": "test"}))

            # Expect the server to echo or respond (optional based on your code)
