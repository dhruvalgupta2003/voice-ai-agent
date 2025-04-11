import json
import base64
import os
from typing import Dict
from fastapi.websockets import WebSocketState
import pywav

from fastapi import FastAPI, WebSocket, Request, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse, Connect
import uvicorn

from config.load_config import Config
from utils.load_prompt import load_prompt
from utils.audio_stream_processor import AudioStreamProcessor
from utils.logger import get_logger

# Load Config
config = Config()

app = FastAPI()

# Setup Logs
logger = get_logger()

# Create global audio processor
audio_processor = AudioStreamProcessor()

# Load system prompt
SYSTEM_MESSAGE = load_prompt('system_prompt')

# Maintain session-wise buffers ---> Store raw µ-law byte chunks
raw_audio_buffers:Dict = {}  # session_id -> list of raw bytes

# ------------ Save Raw Audio to .wav file ------------------------
async def save_raw_audio(session_id):
    if session_id in raw_audio_buffers:
        os.makedirs("recordings", exist_ok=True)
        filepath = f"recordings/{session_id}.wav"

        raw_data = b"".join(raw_audio_buffers[session_id])

        # Write µ-law encoded audio directly using pywav
        wave_write = pywav.WavWrite(filepath, 1, 8000, 8, 7)  # 1: mono, 8000Hz, 8bit, 7: µ-law
        wave_write.write(raw_data)
        wave_write.close()

        logger.info(f"Saved µ-law audio to {filepath}")



@app.get("/", response_class=HTMLResponse)
async def index_page():
    return {"message": "Custom Voice Agent Media Stream Server is running!"}

@app.post("/make-call")
async def make_call(request: Request):
    """Make an outgoing call to the specified phone number."""
    data = await request.json()
    to_phone_number = data.get("to")
    if not to_phone_number:
        return {"error": "Phone number is required"}

    client = Client(config.TWILIO_ACCOUNT_SID, config.TWILIO_AUTH_TOKEN)
    call = client.calls.create(
        url=f"{config.NGROK_URL}/outgoing-call",
        to=to_phone_number,
        from_=config.TWILIO_PHONE_NUMBER
    )
    return {"call_sid": call.sid}

@app.api_route("/outgoing-call", methods=["GET", "POST"])
async def handle_outgoing_call(request: Request):
    """Handle outgoing call and return TwiML response to connect to Media Stream."""
    response = VoiceResponse()
    response.say("Please wait while we connect your call to our custom AI voice assistant...")
    response.pause(length=1)
    response.say("You're now connected. Please start speaking!")
    connect = Connect()
    connect.stream(url=f'wss://{request.url.hostname}/media-stream')
    # connect.stream(url=f'wss://{request.url.hostname}/v1/realtime')
    
    response.append(connect)
    return HTMLResponse(content=str(response), media_type="application/xml")

@app.websocket("/media-stream")
async def handle_media_stream(websocket: WebSocket):
    """Handle WebSocket connections between Twilio and our custom audio pipeline."""
    logger.info("Client connected to media stream")
    session_id = None
    
    try:
        await websocket.accept()
        logger.info("WebSocket connection accepted")
        
        # Create session
        session_id = await audio_processor.create_session(websocket)
        logger.info(f"Session created with ID: {session_id}")
        
        stream_sid = None
        
        # Initialize buffer for this session
        raw_audio_buffers[session_id] = []
        
        async for message in websocket.iter_text():
            try:
                data = json.loads(message)
                if data['event'] == "connected":
                    logger.info('Connected to Twilio Successfully !!!')

                elif data['event'] == 'start':
                    stream_sid = data['start']['streamSid']
                    logger.info(f"Incoming stream has started {stream_sid}")
                    
                    # test audio payload
                    # Read the .wav file and encode it in base64
                    with open('temp_input.wav', "rb") as audio_file:
                        audio_bytes = audio_file.read()
                        audio_payload = base64.b64encode(base64.b64decode(audio_bytes)).decode("utf-8")

                    # Send welcome message
                    audio_delta = {
                        "event": "media",
                        "streamSid": stream_sid,
                        "media": {
                            "payload": audio_payload
                        }
                    }
                    print(audio_delta)
                    await websocket.send_json(audio_delta)
                elif data['event'] == 'media':
                    # Handle media payload
                    payload = base64.b64decode(data['media']['payload'])
                    
                    # Add to buffer
                    raw_audio_buffers[session_id].append(payload)
                    # logger.info(len(raw_audio_buffers[session_id]))
                    # Process when buffer reaches threshold
                    conversation_history = []
                    if len(raw_audio_buffers[session_id]) >= config.CHUNK_THRESHOLD:
                        await save_raw_audio(session_id)
                        transcribed_text = await audio_processor._transcribe_audio(raw_audio_buffers[session_id])
                        
                        # Add user message to history
                        conversation_history.append({"role": "user", "content": transcribed_text})
                    
                        llm_response = await audio_processor._generate_llm_response(system_message=SYSTEM_MESSAGE, conversation=conversation_history)
                        
                        logger.info(llm_response)

                    #     # await audio_processor.handle_media_batch(
                    #     #     session_id, 
                    #     #     raw_audio_buffers[session_id], 
                    #     #     stream_sid
                    #     # )
                    #     await audio_processor._transcribe_audio(raw_audio_buffers[session_id])
                    #     # Clear buffer after processing
                    #     raw_audio_buffers[session_id].clear()
                
                elif data['event'] == 'stop':
                    logger.info(f"Stream {stream_sid} has stopped")
                    # await audio_processor._transcribe_audio(raw_audio_buffers[session_id])
                    
                    
                    # # Process any remaining audio in buffer
                    # if session_id in raw_audio_buffers and raw_audio_buffers[session_id]:
                    #     await audio_processor.handle_media_batch(
                    #         session_id, 
                    #         raw_audio_buffers[session_id], 
                    #         stream_sid
                    #     )
                    
                    # Save audio for later analysis if needed
                    # await save_raw_audio(session_id)
                    logger.info("TRANSCRIBER CLOSED")
                
                else:
                    logger.debug(f"Received unknown event type: {data['event']}")
                    
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON received for session {session_id}")
            except Exception as e:
                logger.error(f"Error processing message for session {session_id}: {e}")
                
    except WebSocketDisconnect:
        logger.info(f"Client disconnected for session {session_id}")
    except Exception as e:
        logger.error(f"Error in WebSocket handler: {e}")
        # Try to send error message to client
        try:
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_json({"type": "error", "message": "Server error occurred"})
        except Exception:
            pass
    finally:
        # Clean up
        if session_id:
            logger.info(f"Cleaning up session {session_id}")
            # Process any remaining audio in buffer
            if session_id in raw_audio_buffers and raw_audio_buffers[session_id]:
                try:
                    await audio_processor.handle_media_batch(
                        session_id, 
                        raw_audio_buffers[session_id], 
                        stream_sid
                    )
                except Exception as e:
                    logger.error(f"Error processing final audio batch: {e}")
                    
            # Save audio for analysis
            try:
                await save_raw_audio(session_id)
            except Exception as e:
                logger.error(f"Error saving raw audio: {e}")
                
            # Close session
            audio_processor.close_session(session_id)
            raw_audio_buffers.pop(session_id, None)
            
        logger.info("WebSocket handler completed")
    
# @app.websocket("/media-stream")
# async def handle_media_stream(websocket: WebSocket):
#     """Handle WebSocket connections between Twilio and our custom audio pipeline."""
#     logger.info("Client connected to media stream")
#     await websocket.accept()
    
#     # Create session for this connection
#     session_id = await audio_processor.create_session(websocket)
#     stream_sid = None
    
#     try:
#         async for message in websocket.iter_text():
#             data = json.loads(message)

#             if data['event'] == 'media':
#                 logger.info(data)
#                 # Decode base64 audio
#                 audio_bytes = base64.b64decode(data['media']['payload'])
#                 # Process through our custom pipeline
#                 await audio_processor.process_audio_chunk(session_id, audio_bytes)
                
#             elif data['event'] == 'start':
#                 logger.info(data)
#                 stream_sid = data['start']['streamSid']
#                 logger.info(f"Incoming stream has started {stream_sid}")
                
#             elif data['event'] == 'stop':
#                 logger.info(data)
#                 logger.info(f"Stream {stream_sid} has stopped")
#     except WebSocketDisconnect:
#         logger.info("Client disconnected")
#     except Exception as e:
#         logger.info(f"Error in WebSocket handler: {e}")
#     finally:
#         # Clean up session
#         audio_processor.close_session(session_id)


if __name__ == "__main__":
    # nosec B104: this host binding is intended in controlled environments
    # trunk-ignore(bandit/B104)
    uvicorn.run(app, host="0.0.0.0", port=config.PORT)