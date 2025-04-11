import os
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Twilio credentials
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE = os.getenv("TWILIO_PHONE")

# AssemblyAI API Key
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")

# Groq API Key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# AWS Credentials
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

# System message for the LLM
SYSTEM_MESSAGE = """
You are a helpful AI voice assistant having a phone conversation with a human. 
Your job is to be friendly, concise, and helpful.

Guidelines:
1. Keep responses brief (1-3 sentences) since this is a voice conversation
2. Be conversational and natural
3. Avoid lengthy explanations or complex language
4. If you don't know something, be honest about it
5. Don't mention that you're an AI in every response
6. Ask clarifying questions when needed
"""

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Initialize API clients
try:
    # Initialize Groq client
    from groq import Groq
    groq_client = Groq(api_key=GROQ_API_KEY)
    
    # Initialize AssemblyAI
    import assemblyai as aai
    aai.settings.api_key = ASSEMBLYAI_API_KEY
    
    # Initialize AWS session
    import boto3
    aws_session = boto3.Session(
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION
    )
    
except ImportError:
    logging.warning("Some dependencies could not be imported. Make sure to install required packages.")
except Exception as e:
    logging.error(f"Error initializing API clients: {e}")
