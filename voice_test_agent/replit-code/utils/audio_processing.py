import os
import logging
import tempfile
import asyncio
import assemblyai as aai
from typing import Union

logger = logging.getLogger(__name__)

async def transcribe_audio(audio_data: Union[bytes, bytearray]) -> str:
    """
    Transcribe audio using AssemblyAI
    
    Args:
        audio_data: The audio data as bytes or bytearray
        
    Returns:
        Transcribed text or "No transcription available" if transcription failed
    """
    try:
        # If audio_data is a bytearray, convert to bytes
        if isinstance(audio_data, bytearray):
            audio_data = bytes(audio_data)
        
        # Check buffer size
        buffer_size = len(audio_data)
        if buffer_size < 1000:  # Skip very small audio chunks
            logger.warning(f"Audio chunk too small ({buffer_size} bytes), skipping transcription")
            return "No transcription available"
            
        # Save to a temporary file with more detailed logging
        temp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                temp_file.write(audio_data)
                temp_file_path = temp_file.name
                logger.info(f"Saved audio to temporary file: {temp_file_path} ({buffer_size} bytes)")
        except Exception as file_error:
            logger.error(f"Failed to create temporary audio file: {file_error}")
            return "No transcription available"
        
        # Check if file was created successfully
        if not os.path.exists(temp_file_path) or os.path.getsize(temp_file_path) == 0:
            logger.error(f"Temporary file creation failed or file is empty: {temp_file_path}")
            return "No transcription available"
            
        logger.info(f"Beginning transcription with AssemblyAI, file size: {os.path.getsize(temp_file_path)} bytes")
        
        # Define transcription function to run in executor
        def run_transcription():
            try:
                # Initialize transcriber with timeout and other settings
                transcriber = aai.Transcriber()
                
                # Transcribe with detailed configuration
                transcript = transcriber.transcribe(
                    temp_file_path,
                    config=aai.TranscriptionConfig(
                        punctuate=True,  # Add punctuation
                        format_text=True,  # Clean up the text
                        language_detection=True  # Auto-detect language
                    )
                )
                
                if transcript and transcript.text:
                    return transcript.text.strip()
                else:
                    return None
            except aai.exceptions.AuthorizationError:
                logger.error("AssemblyAI authorization failed - check your API key")
                return None
            except Exception as e:
                logger.error(f"Error in transcription process: {str(e)}", exc_info=True)
                return None
        
        # Run the transcription in an executor with a timeout
        try:
            # Run transcription in executor
            result = await asyncio.get_event_loop().run_in_executor(None, run_transcription)
        except asyncio.TimeoutError:
            logger.error("Transcription timed out")
            result = None
        except Exception as exec_error:
            logger.error(f"Error running transcription in executor: {exec_error}")
            result = None
        
        # Clean up temporary file
        try:
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                logger.debug(f"Temporary file removed: {temp_file_path}")
        except Exception as cleanup_error:
            logger.warning(f"Error removing temporary file: {cleanup_error}")
        
        # Process the result
        if result:
            logger.info(f"Transcription successful: '{result}'")
            return result
        else:
            logger.warning("Transcription returned no results")
            return "No transcription available"
            
    except Exception as e:
        logger.error(f"Unexpected error in transcribe_audio: {e}", exc_info=True)
        return "No transcription available"
