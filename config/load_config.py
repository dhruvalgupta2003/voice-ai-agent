import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
class Config:
    GROQ_API_KEY = os.getenv('GROQ_API_KEY')
    AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY_ID')
    AWS_SECRET_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
    AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')
    TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
    TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
    TWILIO_PHONE_NUMBER = os.getenv('TWILIO_PHONE_NUMBER')
    ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
    NGROK_URL = os.getenv('NGROK_URL')
    PORT = int(os.getenv('PORT', 8000))
    
    # Chunk Threshold
    CHUNK_THRESHOLD = 32
    
    # TWILIO VOICE
    TWILIO_SAMPLE_RATE = 8000 #hz
    
    # Agent
    AGENT_NAME = 'Ava'
    
    # Company Name (for which ava is working)
    COMPANY_NAME = "OakStone-Mortgages"

    # Constants
    VOICE = 'alloy'
    LOG_EVENT_TYPES = [
    'response.content.done', 'rate_limits.updated', 'response.done',
    'input_audio_buffer.committed', 'input_audio_buffer.speech_stopped',
    'input_audio_buffer.speech_started', 'session.created'
    ]
    
    # OPENAI API KEY
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    # Dir
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    LOGS_DIR = os.path.join(BASE_DIR, 'logs')
