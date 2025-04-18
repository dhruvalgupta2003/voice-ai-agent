import os
import pytest
from unittest.mock import patch
from config.load_config import Config
from utils.validate_config import validate_config

class TestConfig:
    def test_config_loading_success(self):
        # Setup test environment variables
        test_env = {
            "TWILIO_ACCOUNT_SID": "test_sid",
            "TWILIO_AUTH_TOKEN": "test_token",
            "TWILIO_PHONE_NUMBER": "+1234567890",
            "OPENAI_API_KEY": "test_api_key",
            "NGROK_URL": "https://test.ngrok.io",
            "PORT": "8000",
            "AGENT_NAME": "Test Agent",
            "COMPANY_NAME": "Test Company",
            "VOICE": "alloy"
        }
        
        # Use patch to mock environment variables
        with patch.dict(os.environ, test_env):
            config = Config()
            
            # Assert all expected values are loaded
            assert config.TWILIO_ACCOUNT_SID == "test_sid"
            assert config.TWILIO_AUTH_TOKEN == "test_token"
            assert config.TWILIO_PHONE_NUMBER == "+1234567890"
            assert config.OPENAI_API_KEY == "test_api_key"
            assert config.NGROK_URL == "https://test.ngrok.io"
            assert config.PORT == 8000
            assert config.AGENT_NAME == "Test Agent"
            assert config.COMPANY_NAME == "Test Company"
            assert config.VOICE == "alloy"
    
    def test_config_validation_success(self):
        # Create a valid config
        config = Config()
        config.TWILIO_ACCOUNT_SID = "test_sid"
        config.TWILIO_AUTH_TOKEN = "test_token"
        config.TWILIO_PHONE_NUMBER = "+1234567890"
        config.OPENAI_API_KEY = "test_api_key"
        config.NGROK_URL = "https://test.ngrok.io"
        
        # Should not raise exception
        validate_config(config)
    
    def test_config_validation_missing_required(self):
        # Create a config missing required fields
        config = Config()
        config.TWILIO_ACCOUNT_SID = None
        config.TWILIO_AUTH_TOKEN = "test_token"
        config.TWILIO_PHONE_NUMBER = "+1234567890"
        config.OPENAI_API_KEY = "test_api_key"
        
        # Should raise ValueError
        with pytest.raises(ValueError):
            validate_config(config)

    def test_default_values(self):
        # Setup minimal environment variables
        test_env = {
            "TWILIO_ACCOUNT_SID": "test_sid",
            "TWILIO_AUTH_TOKEN": "test_token",
            "TWILIO_PHONE_NUMBER": "+1234567890",
            "OPENAI_API_KEY": "test_api_key",
            "NGROK_URL": "https://test.ngrok.io"
        }
        
        # Use patch to mock environment variables
        with patch.dict(os.environ, test_env):
            config = Config()
            
            # Assert default values for optional configs
            assert config.PORT == 8000  # Default port
            assert config.VOICE == "alloy"  # Default voice
            assert config.AGENT_NAME == "Ava"  # Default agent name
            assert config.COMPANY_NAME == "OakStone-Mortgages"  # Default company name
