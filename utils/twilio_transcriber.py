import assemblyai as aai

from config.load_config import Config
from utils.logger import get_logger

# load config
config = Config()

# setup logging
logger = get_logger()

aai.settings.api_key = config.ASSEMBLYAI_API_KEY

def on_open(session_opened: aai.RealtimeSessionOpened):
    logger.info(f"TRANSCRIBER SESSION ID: {session_opened.session_id}")
    
def on_data(transcript: aai.RealtimeTranscript):
    if not transcript.text:
        return
    
    if isinstance(transcript, aai.RealtimeFinalTranscript):
        logger.info("Transcribed final text....")
        print(transcript.text ,end='\r\n')
    else:
        logger.info("Transcribed partial text...")
        print(transcript.text, end='\r')

def on_error(error: aai.RealtimeError):
    logger.error(f"An Error has occurred {error}")


def on_close():
    logger.info("Closing TRANSCRIBER SESSION <>")
    

class TwilioTranscriber(aai.RealtimeTranscriber):
    def __init__(self):
        super().__init__(
            on_data=on_data,
            on_error=on_error,
            on_open=on_open,
            on_close=on_close,
            sample_rate=config.TWILIO_SAMPLE_RATE,
            encoding=aai.AudioEncoding.pcm_mulaw
        )
        