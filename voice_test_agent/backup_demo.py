import asyncio
from datetime import datetime
from enum import Enum
import json
import base64
import os
from typing import Dict, List
import pywav

from fastapi import FastAPI, WebSocket, Request, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse, Connect
import uvicorn
import websockets

from config.load_config import Config
from utils.load_prompt import load_prompt
from utils.audio_stream_processor import AudioStreamProcessor
from utils.logger import get_logger
from utils.validate_config import validate_config
from utils.generate_call_summary import generate_call_summary_and_tags

# Load Config
config = Config()

# Validate Config
validate_config(config)

app = FastAPI()

# Setup Logs
logger = get_logger()

# Create global audio processor
audio_processor = AudioStreamProcessor()

# Call States for tracking conversation flow
class CallState(str, Enum):
    INTRODUCTION = "introduction"
    VERIFICATION = "verification"
    LOAN_STATUS = "loan_status"
    PAYMENT_OPTION = "payment_option"
    PAYMENT_PROCESS = "payment_process"
    PAYMENT_LATER = "payment_later"
    HARDSHIP = "hardship"
    DISPUTE = "dispute"
    SATISFACTION = "satisfaction"
    COMPLETED = "completed"

# Call outcome tags
class CallOutcomeTag(str, Enum):
    PAYMENT_COMPLETED = "payment_completed"
    PAYMENT_SCHEDULED = "payment_scheduled"
    REQUIRES_FOLLOWUP = "requires_followup"
    FINANCIAL_HARDSHIP = "financial_hardship"
    DISPUTED_LOAN = "disputed_loan"
    WRONG_NUMBER = "wrong_number"
    UNSUCCESSFUL = "unsuccessful"
    HIGH_SATISFACTION = "high_satisfaction"
    LOW_SATISFACTION = "low_satisfaction"

# Call data storage
call_data:Dict = {}

# Load system prompt
SYSTEM_MESSAGE = load_prompt('loan_repayment_prompt')

# Maintain session-wise buffers ---> Store raw µ-law byte chunks
raw_audio_buffers:Dict = {}  # session_id -> list of raw bytes

# Store transcripts for summary generation
call_transcripts: Dict[str, List[dict]] = {}  # session_id -> list of conversation turns

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

# ---------- Store call Data ------------------
# Store call data (summary and tags)
async def store_call_data(session_id: str):
    if session_id in call_transcripts and call_transcripts[session_id]:
        summary, tags = generate_call_summary_and_tags(call_transcripts[session_id])
        
        # Create call record
        call_record = {
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "summary": summary,
            "tags": [tag.value for tag in tags],
            "transcript": call_transcripts[session_id]
        }
        
        # In a production environment, you would store this in a database
        # For now, we'll save it to a JSON file
        os.makedirs("call_records", exist_ok=True)
        with open(f"call_records/{session_id}.json", "w") as f:
            json.dump(call_record, f, indent=2)
        
        logger.info(f"Stored call data for session {session_id}")
        return call_record
    return None

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
        
        async with websockets.connect(
            'wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-12-17',
            additional_headers={
                "Authorization": f"Bearer {config.OPENAI_API_KEY}",
                "OpenAI-Beta": "realtime=v1"
            }
        ) as openai_ws:
            logger.info('Connected to OPENAI WEBSOCKET Successfully....')
            await send_session_update(openai_ws)
            stream_sid = None
            session_id = None
            
            # Flag to track if the connection is still valid
            connection_active = True

            async def receive_from_twilio():
                """Receive audio data from Twilio and send it to the OpenAI Realtime API."""
                nonlocal stream_sid, connection_active
                try:
                    async for message in websocket.iter_text():
                        data = json.loads(message)
                        if data['event'] == 'media' and connection_active:
                            audio_append = {
                                "type": "input_audio_buffer.append",
                                "audio": data['media']['payload']
                            }
                            try:
                                await openai_ws.send(json.dumps(audio_append))
                            except Exception as e:
                                logger.error(f"Failed to send to OpenAI: {e}")
                                connection_active = False
                        elif data['event'] == 'start':
                            stream_sid = data['start']['streamSid']
                            print(f"Incoming stream has started {stream_sid}")
                        elif data['event'] == 'stop':
                            logger.info('Twilio Stopped....')
                except WebSocketDisconnect: 
                    print("Client disconnected.")
                    connection_active = False
                    try:
                        await openai_ws.close()
                    except Exception as e:
                        logger.error(f"error in twilio websocket {e}")

            async def send_to_twilio():
                """Receive events from the OpenAI Realtime API, send audio back to Twilio."""
                nonlocal stream_sid, session_id
                try:
                    async for openai_message in openai_ws:
                        response = json.loads(openai_message)
                        if response['type'] in config.LOG_EVENT_TYPES:
                            print(f"Received event: {response['type']}", response)
                        if response['type'] == 'session.created':
                            session_id = response['session']['id']
                        if response['type'] == 'session.updated':
                            print("Session updated successfully:", response)
                        if response['type'] == 'response.audio.delta' and response.get('delta'):
                            try:
                                audio_payload = base64.b64encode(base64.b64decode(response['delta'])).decode('utf-8')
                                audio_delta = {
                                    "event": "media",
                                    "streamSid": stream_sid,
                                    "media": {
                                        "payload": audio_payload
                                    }
                                }
                                await websocket.send_json(audio_delta)
                            except Exception as e:
                                print(f"Error processing audio data: {e}")
                        if response['type'] == 'conversation.item.created':
                            print(f"conversation.item.created event: {response}")
                except Exception as e:
                    print(f"Error in send_to_twilio: {e}")

            await asyncio.gather(receive_from_twilio(), send_to_twilio())
    except Exception as e:
        logger.error(f"Error Occurred {e}")

async def send_session_update(openai_ws):
    """Send session update to OpenAI WebSocket."""
    session_update = {
        "type": "session.update",
        "session": {
            "input_audio_format": "g711_ulaw",
            "output_audio_format": "g711_ulaw",
            "voice": config.VOICE,
            "instructions": SYSTEM_MESSAGE,
            "modalities": ["text", "audio"],
            "temperature": 0.8,
        }
    }
    print('Sending session update:', json.dumps(session_update))
    await openai_ws.send(json.dumps(session_update))
    logger.info("WebSocket handler completed")


if __name__ == "__main__":
    # nosec B104: this host binding is intended in controlled environments
    # trunk-ignore(bandit/B104)
    uvicorn.run(app, host="0.0.0.0", port=config.PORT)