import os
import base64
import logging
import asyncio
import boto3
from typing import Optional

logger = logging.getLogger(__name__)

# AWS credentials from environment variables
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

# Create AWS session
aws_session = boto3.Session(
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION
)

# Create Polly client
polly_client = aws_session.client("polly")

async def text_to_speech(text: str) -> Optional[str]:
    """
    Convert text to speech using AWS Polly
    
    Args:
        text: The text to convert to speech
        
    Returns:
        Base64-encoded audio data or None if conversion failed
    """
    try:
        logger.info(f"Converting to speech: {text}")
        
        # Ensure output directory exists
        os.makedirs('output_dir', exist_ok=True)
        
        # Run Polly in a separate thread since it's synchronous
        def run_polly():
            try:
                response = polly_client.synthesize_speech(
                    Text=text,
                    OutputFormat='mp3',
                    VoiceId='Ruth',  # Neural voice
                    Engine="neural"
                )
                
                if "AudioStream" in response:
                    # Read the audio stream
                    audio_stream = response['AudioStream'].read()
                    
                    # Save the MP3 file for debugging if needed
                    output_audio_path = os.path.join('output_dir', 'output_audio.mp3')
                    with open(output_audio_path, 'wb') as f:
                        f.write(audio_stream)
                    
                    # Base64 encode for WebSocket
                    base64_audio = base64.b64encode(audio_stream).decode('utf-8')
                    logger.info(f"Speech conversion successful, size: {len(base64_audio)}")
                    
                    return base64_audio
                else:
                    logger.error("No AudioStream in Polly response")
                    return None
            except Exception as e:
                logger.error(f"Error in Polly thread: {e}")
                return None
        
        # Run Polly in the executor
        result = await asyncio.get_event_loop().run_in_executor(None, run_polly)
        return result
        
    except Exception as e:
        logger.error(f"Error converting text to speech: {e}", exc_info=True)
        return None
