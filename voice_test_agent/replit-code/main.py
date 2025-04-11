import os
import json
import base64
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import Optional
import asyncio

# Import our utility modules
from config import (
    TWILIO_ACCOUNT_SID, 
    TWILIO_AUTH_TOKEN, 
    SYSTEM_MESSAGE
)
from utils.audio_processing import transcribe_audio
from utils.llm import get_llm_response
from utils.tts import text_to_speech

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Voice AI Agent")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def read_root():
    """Root endpoint to check if the service is running"""
    return {"status": "online", "service": "Voice AI Agent"}

@app.post("/make-call")
async def make_call(phone_number: Optional[str] = None):
    """Initiate an outbound call to the specified phone number with media streaming"""
    if not phone_number:
        try:
            from fastapi import Request
            request = Request
            body = await request.json()
            phone_number = body.get('phone_number')
        except Exception:
            phone_number = ''

    # Validate phone number
    if not phone_number:
        raise HTTPException(status_code=400, detail="Phone number is required")

    try:
        from twilio.rest import Client
        from twilio.twiml.voice_response import VoiceResponse, Connect
        
        # Create TwiML with stream configuration
        twiml = VoiceResponse()
        twiml.say("Connecting to AI voice assistant.")
        twiml.pause(length=1)
        
        # Add stream configuration
        connect = Connect()
        # Use the WEBHOOK_HOST environment variable
        host = os.getenv('WEBHOOK_HOST')
        if not host:
            logger.error("WEBHOOK_HOST environment variable is not set")
            raise HTTPException(status_code=500, detail="Server configuration error: WEBHOOK_HOST not set")
        
        # Log the webhook URL for debugging
        webhook_url = f'wss://{host}/media-stream'
        logger.info(f"Using webhook URL: {webhook_url}")
        connect.stream(url=webhook_url)
        twiml.append(connect)
        
        # Initialize Twilio client
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        
        # Make the call with dynamically generated TwiML
        call = client.calls.create(
            to=phone_number,
            from_=os.getenv('TWILIO_PHONE'),
            twiml=str(twiml)
        )
        logger.info(f"Initiated call to {phone_number} with SID: {call.sid}")
        
        return {"status": "success", "call_sid": call.sid}
    except Exception as e:
        logger.error(f"Error making outbound call: {e}")
        return {"status": "error", "message": str(e)}

@app.websocket("/media-stream")
async def handle_media_stream(websocket: WebSocket):
    """Handle WebSocket connections for the media stream"""
    await websocket.accept()
    logger.info("Client connected to WebSocket")
    
    stream_sid = None
    audio_buffer = bytearray()
    buffer_size_threshold = 8000  # Reduced buffer size for faster processing
    last_process_time = 0.0
    processing_interval = 1.5  # Process every 1.5 seconds
    last_media_time = asyncio.get_event_loop().time()
    media_timeout = 5.0  # Consider connection stale after 5 seconds without media

    try:
        # Raise log level for this function to see more details
        logging.getLogger(__name__).setLevel(logging.DEBUG)
        logger.debug("Media stream handler started with DEBUG logging")
        
        # Send initial connection confirmation
        try:
            await websocket.send_json({
                "event": "connected",
                "message": "WebSocket connection established"
            })
            logger.info("Sent connection confirmation")
        except Exception as e:
            logger.error(f"Failed to send connection confirmation: {e}")
        
        while True:
            try:
                # Use a shorter timeout to periodically check connection status
                message = await asyncio.wait_for(
                    websocket.receive_text(), 
                    timeout=1.0
                )
                
                # Parse message
                data = json.loads(message)
                current_time = asyncio.get_event_loop().time()
                
                # Process different event types
                if data['event'] == 'start':
                    stream_sid = data['start']['streamSid']
                    logger.info(f"Stream started: {stream_sid}")
                    
                    # Send a welcome message
                    welcome_message = "Hello! I'm your AI voice assistant. How can I help you today?"
                    logger.info("Generating welcome message audio")
                    audio_payload = await text_to_speech(welcome_message)
                    if audio_payload:
                        logger.info(f"Sending welcome audio, size: {len(audio_payload)}")
                        await websocket.send_json({
                            "event": "media",
                            "streamSid": stream_sid,
                            "media": {
                                "payload": audio_payload
                            }
                        })
                        logger.info("Welcome message sent successfully")
                    else:
                        logger.error("Failed to generate welcome audio")
                    
                elif data['event'] == 'media':
                    # Update the last media time
                    last_media_time = current_time
                    
                    # More detailed logging
                    payload_size = len(data['media']['payload'])
                    logger.debug(f"Received media chunk, size: {payload_size} bytes")
                    
                    # Accumulate audio data
                    try:
                        audio_chunk = base64.b64decode(data['media']['payload'])
                        audio_buffer.extend(audio_chunk)
                        logger.debug(f"Buffer size after adding chunk: {len(audio_buffer)} bytes")
                    except Exception as e:
                        logger.error(f"Error decoding audio chunk: {e}")
                    
                    # Check if we should process based on buffer size or time interval
                    should_process = (
                        len(audio_buffer) >= buffer_size_threshold or
                        (current_time - last_process_time) >= processing_interval and len(audio_buffer) > 1000
                    )
                    
                    if should_process:
                        logger.info(f"Processing audio buffer, size: {len(audio_buffer)} bytes")
                        
                        # Process in background task to avoid blocking the WebSocket
                        process_task = asyncio.create_task(
                            process_audio_buffer(websocket, stream_sid, audio_buffer)
                        )
                        
                        # Clear buffer and reset timer
                        audio_buffer = bytearray()
                        last_process_time = current_time
                    
                elif data['event'] == 'stop':
                    logger.info("Stream stop event received")
                    # Process any remaining audio
                    if len(audio_buffer) > 0:
                        logger.info(f"Processing final audio buffer, size: {len(audio_buffer)} bytes")
                        await process_audio_buffer(websocket, stream_sid, audio_buffer)
                    
                    logger.info("Sending final message")
                    # Send a goodbye message
                    goodbye_message = "Thank you for calling. Goodbye!"
                    audio_payload = await text_to_speech(goodbye_message)
                    if audio_payload and stream_sid:
                        await websocket.send_json({
                            "event": "media",
                            "streamSid": stream_sid,
                            "media": {
                                "payload": audio_payload
                            }
                        })
                    break
                
                # Check if we haven't received media for a while, but still want to process existing buffer
                if (current_time - last_media_time) > media_timeout and len(audio_buffer) > 1000:
                    logger.warning(f"No media received for {current_time - last_media_time:.1f}s, processing existing buffer")
                    await process_audio_buffer(websocket, stream_sid, audio_buffer)
                    audio_buffer = bytearray()
                    last_process_time = current_time
                
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON received: {e}")
                continue
                
    except asyncio.TimeoutError:
        # No new data for a while, check if we have audio to process
        current_time = asyncio.get_event_loop().time()
        if len(audio_buffer) > 1000:
            logger.info(f"Processing audio after timeout, buffer size: {len(audio_buffer)} bytes")
            await process_audio_buffer(websocket, stream_sid, audio_buffer)
            audio_buffer = bytearray()
            last_process_time = current_time
        
        # Check if the media stream is still active
        if (current_time - last_media_time) > media_timeout:
            logger.warning(f"No media for {current_time - last_media_time:.1f}s, connection may be stale")
            
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected by client")
    except Exception as e:
        logger.error(f"Error in media stream handling: {str(e)}", exc_info=True)
    finally:
        logger.info("Closing WebSocket connection")
        try:
            await websocket.close()
        except Exception as e:
            logger.error(f"Error closing WebSocket: {e}")
        logger.info("WebSocket connection closed")

async def process_audio_buffer(websocket, stream_sid, audio_buffer):
    """Process the accumulated audio buffer"""
    if not stream_sid:
        logger.warning("No stream SID available, cannot process audio")
        return
        
    if len(audio_buffer) < 1000:
        logger.warning(f"Audio buffer too small ({len(audio_buffer)} bytes), skipping processing")
        return

    try:
        # Create a copy of the buffer to avoid reference issues
        buffer_copy = audio_buffer.copy()
        
        # Track processing time for performance monitoring
        start_time = asyncio.get_event_loop().time()
        
        # Log detailed buffer info
        logger.debug(f"Processing audio buffer: size={len(buffer_copy)} bytes, stream_sid={stream_sid}")
        
        # Step 1: Transcribe audio
        logger.info("Transcribing audio...")
        transcript = await transcribe_audio(buffer_copy)
        transcribe_time = asyncio.get_event_loop().time() - start_time
        logger.debug(f"Transcription took {transcribe_time:.2f} seconds")
        
        # Check transcription result
        if not transcript:
            logger.warning("Transcription failed or returned None")
            return
            
        if transcript == "No transcription available":
            logger.info("No speech detected in audio")
            return
        
        # Log the successful transcription
        logger.info(f"Transcribed text: {transcript}")
        
        # Step 2: Get LLM response
        logger.info("Getting LLM response...")
        llm_start_time = asyncio.get_event_loop().time()
        llm_response = await get_llm_response(transcript, SYSTEM_MESSAGE)
        llm_time = asyncio.get_event_loop().time() - llm_start_time
        logger.debug(f"LLM processing took {llm_time:.2f} seconds")
        
        if not llm_response:
            logger.warning("LLM response failed or returned None")
            # Send a fallback message
            llm_response = "I'm sorry, I didn't catch that. Could you please repeat?"
        
        logger.info(f"LLM response: {llm_response}")
        
        # Step 3: Convert response to speech
        logger.info("Converting to speech...")
        tts_start_time = asyncio.get_event_loop().time()
        audio_payload = await text_to_speech(llm_response)
        tts_time = asyncio.get_event_loop().time() - tts_start_time
        logger.debug(f"Text-to-speech took {tts_time:.2f} seconds")
        
        if not audio_payload:
            logger.error("Failed to convert response to speech")
            return
            
        # Step 4: Send response back to the call
        logger.info(f"Sending audio response back to call (size: {len(audio_payload)} bytes)...")
        try:
            await websocket.send_json({
                "event": "media",
                "streamSid": stream_sid,
                "media": {
                    "payload": audio_payload
                }
            })
            logger.info("Audio response sent successfully")
            
            # Log total processing time
            total_time = asyncio.get_event_loop().time() - start_time
            logger.info(f"Total processing time: {total_time:.2f} seconds")
            
        except Exception as send_error:
            logger.error(f"Failed to send audio response: {send_error}")
            
    except Exception as e:
        logger.error(f"Error processing audio buffer: {e}", exc_info=True)

if __name__ == "__main__":
    # Create the output directory for audio files
    os.makedirs('output_dir', exist_ok=True)
    
    # Run the FastAPI app with Uvicorn
    port = int(os.getenv("PORT", "8000"))
    logger.info(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
