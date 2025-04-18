import pytest
from unittest.mock import patch, MagicMock
from fastapi import HTTPException

class TestToolExecution:
    @pytest.mark.asyncio
    @patch('demo.twilio_client')
    async def test_send_payment_link_valid(self, mock_twilio):
        from demo import send_payment_link
        
        # Mock Twilio message creation
        mock_message = MagicMock()
        mock_message.sid = "test_message_sid"
        mock_twilio.messages.create.return_value = mock_message
        
        # Test with valid parameters
        result = await send_payment_link(
            phone_number="+1234567890",
            amount=100.50,
            call_id="test_call_id"
        )
        
        # Verify result
        assert result["success"] is True
        assert "msg_sid" in result
        assert result["msg_sid"] == "test_message_sid"
        assert "link" in result
    
    @pytest.mark.asyncio
    async def test_get_payment_status_valid(self):
        from demo import get_payment_status
        
        # Create test payment status file
        import os
        import json
        os.makedirs("payment_status", exist_ok=True)
        test_status = {"status": True, "amount": "$100.50"}
        with open("payment_status/test_call_id.json", "w") as f:
            json.dump(test_status, f)
        
        # Test with valid call_id
        result = await get_payment_status("test_call_id")
        
        # Verify result
        assert result["status"] is True
        assert result["amount"] == "$100.50"
        
        # Clean up
        os.remove("payment_status/test_call_id.json")
    
    @pytest.mark.asyncio
    async def test_get_payment_status_not_found(self):
        from demo import get_payment_status
        
        # Test with non-existent call_id
        with pytest.raises(HTTPException) as exc_info:
            await get_payment_status("nonexistent_id")
        
        # Verify exception
        assert exc_info.value.status_code == 404
    
    @pytest.mark.asyncio
    @patch('demo.send_payment_link')
    async def test_execute_tool_send_payment_link(self, mock_send_payment_link):
        from demo import execute_tool
        
        # Mock send_payment_link
        mock_send_payment_link.return_value = {
            "success": True,
            "message": "Payment link sent successfully",
            "msg_sid": "test_message_sid",
            "link": "https://test.link"
        }
        
        # Test with valid parameters
        result = await execute_tool(
            tool_name="send_payment_link",
            parameters={
                "phone_number": "+1234567890",
                "amount": 100.50,
                "call_id": "test_call_id"
            }
        )
        
        # Verify result
        assert result["success"] is True
        assert "message" in result
        assert "details" in result
    
    @pytest.mark.asyncio
    @patch('demo.get_payment_status')
    async def test_execute_tool_get_payment_status(self, mock_get_payment_status):
        from demo import execute_tool
        
        # Mock get_payment_status
        mock_get_payment_status.return_value = {
            "status": True,
            "amount": "$100.50"
        }
        
        # Test with valid parameters
        result = await execute_tool(
            tool_name="get_payment_status",
            parameters={
                "call_id": "test_call_id"
            }
        )
        
        # Verify result
        assert result["success"] is True
        assert "message" in result
        assert "status" in result
    
    @pytest.mark.asyncio
    async def test_execute_tool_unknown(self):
        from demo import execute_tool
        
        # Test with unknown tool
        result = await execute_tool(
            tool_name="unknown_tool",
            parameters={}
        )
        
        # Verify result
        assert result["success"] is False
        assert "message" in result
        assert "Unknown tool" in result["message"]
