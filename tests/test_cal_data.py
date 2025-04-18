import pytest
import os
from unittest.mock import patch

class TestCallData:
    @pytest.mark.asyncio
    @patch('demo.generate_call_summary_and_tags')
    @patch('demo.save_raw_audio')
    @patch('demo.transcribe_user_call_record')
    @patch('demo.rewrite_call_record_with_user_conversation')
    async def test_store_call_data(self, mock_rewrite, mock_transcribe, mock_save_audio, mock_generate_summary):
        from demo import store_call_data, call_transcripts
        
        # Mock functions
        mock_save_audio.return_value = True
        mock_generate_summary.return_value = ("Test summary", ["payment_completed", "high_satisfaction"])
        mock_transcribe.return_value = ["Hello", "I want to pay my loan"]
        mock_rewrite.return_value = "Rewrite completed"
        
        # Create test transcript
        session_id = "test_session_id"
        call_id = "test_call_id"
        call_transcripts[session_id] = [
            {"role": "assistant", "content": "Hello, how can I help?"},
            {"role": "user", "content": "I want to pay my loan"}
        ]
        
        # Create test directory
        os.makedirs("recordings", exist_ok=True)
        os.makedirs("call_records", exist_ok=True)
        
        # Create dummy audio file
        with open(f"recordings/{session_id}.wav", "wb") as f:
            f.write(b"test audio data")
        
        # Test store_call_data
        result = await store_call_data(session_id, call_id)
        
        # Verify result
        assert result is not None
        assert result["call_id"] == call_id
        assert result["session_id"] == session_id
        assert result["summary"] == "Test summary"
        
        # Verify file was created
        assert os.path.exists(f"call_records/{call_id}.json")
        
        # Clean up
        os.remove(f"recordings/{session_id}.wav")
        os.remove(f"call_records/{call_id}.json")
        if session_id in call_transcripts:
            del call_transcripts[session_id]
    
    @pytest.mark.asyncio
    async def test_store_call_data_no_transcript(self):
        from demo import store_call_data, call_transcripts
        
        # Create test data
        session_id = "test_session_id_empty"
        call_id = "test_call_id_empty"
        
        # Don't add any transcripts
        call_transcripts[session_id] = []
        
        # Test store_call_data with empty transcript
        result = await store_call_data(session_id, call_id)
        
        # Verify result
        assert result is None
        
        # Clean up
        if session_id in call_transcripts:
            del call_transcripts[session_id]
