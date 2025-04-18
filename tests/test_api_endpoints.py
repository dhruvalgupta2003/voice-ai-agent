import json
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from demo import app

client = TestClient(app)

class TestApiEndpoints:
    def test_index_route(self):
        response = client.get("/")
        assert response.status_code == 200
        json_data = response.json()
        assert "message" in json_data
        assert isinstance(json_data["message"], str)
        assert json_data["message"] == "Custom Voice Agent Media Stream Server is running!"  # if applicable
    
    @patch('demo.twilio_client')
    def test_make_call_valid(self, mock_twilio):
        # Mock Twilio call creation
        mock_call = MagicMock()
        mock_call.sid = "test_call_sid"
        mock_twilio.calls.create.return_value = mock_call
        
        # Test with valid parameters
        response = client.post(
            "/make-call",
            json={"to": "+1234567890"}
        )
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert "call_sid" in data
        assert "call_id" in data
        assert data["call_sid"] == "test_call_sid"
    
    @patch('demo.twilio_client')
    def test_make_call_with_customer_data(self, mock_twilio):
        # Mock Twilio call creation
        mock_call = MagicMock()
        mock_call.sid = "test_call_sid"
        mock_twilio.calls.create.return_value = mock_call
        
        # Test with custom customer data
        response = client.post(
            "/make-call",
            json={
                "to": "+1234567890",
                "customer_name": "Jane Doe",
                "account_number": "9876",
                "loan_amount": "5000",
                "due_date": "2025-05-01"
            }
        )
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert "call_sid" in data
        assert "call_id" in data
    
    @patch('demo.twilio_client')
    def test_end_call_valid(self, mock_twilio):
        # Mock Twilio call update
        mock_twilio.calls.return_value.update.return_value = None
        
        # Test with valid call_sid
        response = client.post("/end-call/test_call_sid")
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
    
    @patch('demo.twilio_client')
    def test_end_call_invalid(self, mock_twilio):
        # Mock Twilio call update to raise exception
        mock_twilio.calls.return_value.update.side_effect = Exception("Call not found")
        
        # Test with invalid call_sid
        response = client.post("/end-call/invalid_sid")
        
        # Verify response
        assert response.status_code == 404
    
    @patch('demo.twilio_client')
    def test_send_payment_link_valid(self, mock_twilio):
        # Mock Twilio message creation
        mock_message = MagicMock()
        mock_message.sid = "test_message_sid"
        mock_twilio.messages.create.return_value = mock_message
        
        # Test with valid parameters
        response = client.post(
            "/send-payment-link",
            json={
                "phone_number": "+1234567890",
                "amount": 100.50,
                "call_id": "test_call_id"
            }
        )
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "msg_sid" in data
        assert "link" in data
    
    def test_send_payment_link_missing_phone(self):
        # Test with missing phone number
        response = client.post(
            "/send-payment-link",
            json={
                "amount": 100.50,
                "call_id": "test_call_id"
            }
        )
        
        # Verify response
        assert response.status_code == 422  # Validation error
    
    def test_payment_status_update_valid(self):
        # Test with valid parameters
        response = client.post(
            "/payment-status",
            json={
                "phone": "+1234567890",
                "call_id": "test_call_id",
                "amount": 100.50
            }
        )
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        
        # Verify the file was created
        import os
        assert os.path.exists("payment_status/test_call_id.json")
        
        # Clean up
        os.remove("payment_status/test_call_id.json")
    
    def test_payment_status_update_missing_fields(self):
        # Test with missing fields
        response = client.post(
            "/payment-status",
            json={
                "phone": "+1234567890",
                # Missing call_id and amount
            }
        )
        
        # Verify response
        assert response.status_code == 400
    
    def test_get_payment_status_valid(self):
        # Create a test payment status file
        import os
        os.makedirs("payment_status", exist_ok=True)
        with open("payment_status/test_call_id.json", "w") as f:
            json.dump({"status": True, "amount": "$100.50"}, f)
        
        # Test with valid call_id
        response = client.get("/payment-status/test_call_id")
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["status"] is True
        assert data["amount"] == "$100.50"
        
        # Clean up
        os.remove("payment_status/test_call_id.json")
    
    def test_get_payment_status_not_found(self):
        # Test with non-existent call_id
        response = client.get("/payment-status/nonexistent_id")
        
        # Verify response
        assert response.status_code == 404
    
    def test_get_call_summary_valid(self):
        # Create a test call record file
        import os
        os.makedirs("call_records", exist_ok=True)
        test_record = {
            "call_id": "test_call_id",
            "session_id": "test_session_id",
            "timestamp": "2023-01-01T12:00:00",
            "summary": "Test call summary",
            "tags": ["payment_completed"],
            "transcript": [{"role": "assistant", "content": "Hello"}]
        }
        with open("call_records/test_call_id.json", "w") as f:
            json.dump(test_record, f)
        
        # Test with valid call_id
        response = client.get("/call-summary/test_call_id")
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["call_id"] == "test_call_id"
        assert data["summary"] == "Test call summary"
        
        # Clean up
        os.remove("call_records/test_call_id.json")
    
    def test_get_call_summary_not_found(self):
        # Test with non-existent call_id
        response = client.get("/call-summary/nonexistent_id")
        
        # Verify response
        assert response.status_code == 200  # Returns 200 even if not found
        data = response.json()
        assert "error" in data

    def test_demo_pay_valid(self):
        # Test with valid parameters
        response = client.get(
            "/demo-pay",
            params={
                "phone": "+1234567890",
                "amount": 100.50,
                "call_id": "test_call_id"
            }
        )
        
        # Verify response
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        
        # Check HTML content
        html_content = response.text
        assert "+1234567890" in html_content
        assert "100.50" in html_content
        assert "test_call_id" in html_content
        assert "Pay Now" in html_content
