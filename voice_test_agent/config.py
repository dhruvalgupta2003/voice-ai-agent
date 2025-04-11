import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    TWILIO_ACCOUNT_SID=os.getenv('TWILIO_ACCOUNT_SID')
    TWILIO_AUTH_TOKEN=os.getenv('TWILIO_AUTH_TOKEN')
    TWILIO_PHONE_NUMBER=os.getenv('TWILIO_PHONE')
    GROQ_API_KEY=os.getenv('GROQ_API_KEY')
    ASSEMBLYAI_API_KEY=os.getenv('ASSEMBLYAI_API_KEY')
    AWS_ACCESS_KEY_ID=os.getenv('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY=os.getenv('AWS_SECRET_ACCESS_KEY')
    AWS_REGION=os.getenv('AWS_REGION')