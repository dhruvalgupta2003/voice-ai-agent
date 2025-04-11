import asyncio
from datetime import datetime
from enum import Enum
import json
import base64
import os
from typing import Dict, List
import uuid
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

# Initialize Twilio Client
twilio_client = Client(config.TWILIO_ACCOUNT_SID, config.TWILIO_AUTH_TOKEN)

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
        summary, tags = await generate_call_summary_and_tags(call_transcripts[session_id])
        
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
    
    # Get loan account details - in production this would come from your CRM/database
    customer_name = data.get("customer_name", "Michael")
    account_number = data.get("account_number", "1234")
    loan_amount = data.get("loan_amount", "$10,000")
    due_date = data.get("due_date", "15th April 2025")
    
    parsed_due_date = datetime.strptime(due_date, "%dth %B %Y").date()

    # Determine overdue or upcoming
    today = datetime.today().date()
    overdue_or_upcoming = "Overdue" if parsed_due_date < today else "Upcoming"

    if not to_phone_number:
        return {"error": "Phone number is required"}

    # Store call context for this call
    call_context = {
        "customer_name": customer_name,
        "account_number": account_number,
        "loan_amount": loan_amount,
        "due_date": due_date,
        "state": CallState.INTRODUCTION
    }
    
    call_id = str(uuid.uuid4())
    call_data[call_id] = call_context
    
    # Create customized system prompt with call details
    custom_prompt = SYSTEM_MESSAGE.replace("[Customer Name]", customer_name)
    custom_prompt = custom_prompt.replace("XXXX", account_number[-4:] if len(account_number) >= 4 else account_number)
    custom_prompt = custom_prompt.replace("₹XX,XXX", f"${loan_amount}")
    custom_prompt = custom_prompt.replace("[Date]", due_date)
    custom_prompt = custom_prompt.replace("[Agent Name]", config.AGENT_NAME)
    custom_prompt = custom_prompt.replace("[Company Name]", config.COMPANY_NAME)
    custom_prompt = custom_prompt.replace("[Overdue/Upcoming in X days]", overdue_or_upcoming)

    # Store the custom prompt for this call
    call_data[call_id]["system_prompt"] = custom_prompt
    

    call = twilio_client.calls.create(
        url=f"{config.NGROK_URL}/outgoing-call?call_id={call_id}",
        to=to_phone_number,
        from_=config.TWILIO_PHONE_NUMBER
    )
    
    # Store the Twilio call SID for reference
    call_data[call_id]["call_sid"] = call.sid
    
    return {"call_sid": call.sid, "call_id": call_id}


# @app.api_route("/outgoing-call", methods=["GET", "POST"])
# async def handle_outgoing_call(request: Request):
#     """Handle outgoing call and return TwiML response to connect to Media Stream."""
#    # Get call_id from query parameters
#     params = dict(request.query_params)
#     call_id = params.get("call_id", "")
#     print('outgoing call call id ', call_id)
#     response = VoiceResponse()
#     response.say("Connecting your call to our loan collection assistant...")
#     response.pause(length=1)
    
#     connect = Connect()
#     connect.stream(url=f'wss://{request.url.hostname}/media-stream?call_id={call_id}')
    
#     response.append(connect)
#     return HTMLResponse(content=str(response), media_type="application/xml")
@app.api_route("/outgoing-call", methods=["GET", "POST"])
async def handle_outgoing_call(request: Request):
    """Handle outgoing call and return TwiML response to connect to Media Stream."""
    # Get call_id from query parameters
    params = dict(request.query_params)
    call_id = params.get("call_id", "abcd")
    print('outgoing call call id ', call_id)
    
    response = VoiceResponse()
    response.say("Connecting your call to our loan collection assistant...")
    response.pause(length=1)
    
    connect = Connect()
    # The issue is here - WebSocket URL needs to be absolute, not relative
    connect.stream(url=f'{config.NGROK_URL.replace("http:", "ws:").replace("https:", "wss:")}/media-stream/{call_id}')
    
    response.append(connect)
    return HTMLResponse(content=str(response), media_type="application/xml")

@app.get("/call-summary/{call_id}")
async def get_call_summary(call_id: str):
    """API endpoint to retrieve call summary and tags."""
    try:
        # In production, you would retrieve this from your database
        with open(f"call_records/{call_id}.json", "r") as f:
            call_record = json.load(f)
        return call_record
    except FileNotFoundError:
        return {"error": "Call record not found"}

@app.websocket("/media-stream/{call_id}")
async def handle_media_stream(websocket: WebSocket, call_id:str):
    """Handle WebSocket connections between Twilio and our custom audio pipeline."""
    logger.info("Client connected to media stream")
    
    session_id = None
    
    try:
        await websocket.accept()
        logger.info("WebSocket connection accepted")
        
        # Create session
        session_id = await audio_processor.create_session(websocket)
        logger.info(f"Session created with ID: {session_id}")
        
        # Associate this session with the call_id if provided
        if call_id and call_id in call_data:
            call_data[call_id]["session_id"] = session_id
        
        stream_sid = None
        
        # Initialize buffer for this session
        raw_audio_buffers[session_id] = []
        
        # Initialize transcript for this session
        call_transcripts[session_id] = []
        
        # Get custom prompt if available
        # custom_prompt = call_data[call_id]["system_prompt"]
        if call_id and call_id in call_data and "system_prompt" in call_data[call_id]:
            custom_prompt = call_data[call_id]["system_prompt"]

        async with websockets.connect(
            'wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-12-17',
            additional_headers={
                "Authorization": f"Bearer {config.OPENAI_API_KEY}",
                "OpenAI-Beta": "realtime=v1"
            }
        ) as openai_ws:
            logger.info('Connected to OPENAI WEBSOCKET Successfully....')
            await send_session_update(openai_ws, custom_prompt)
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
                            if session_id:
                                await store_call_data(session_id)
                                await save_raw_audio(session_id)
                except WebSocketDisconnect: 
                    print("Client disconnected.")
                    connection_active = False
                    # Call ended, generate summary and store data
                    if session_id:
                        await store_call_data(session_id)
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
                        
                        # store session ID
                        if response['type'] == 'session.created':
                            session_id = response['session']['id']
                        
                        if response['type'] == 'session.updated':
                            print("Session updated successfully:", response)
                        
                        # Handle transcripts for summary
                        if response['type'] == 'conversation.item.created':
                            print(f"conversation.item.created event: {response}")
                            # Store conversation transcript
                            if 'item' in response and 'content' in response['item']:
                                if 'text' in response['item']['content']:
                                    call_transcripts[session_id].append({
                                        'role': response['item']['role'],
                                        'content': response['item']['content']['text']
                                    })
                        if response.get("type") == "response.done":
                            try:
                                inner_response = response.get("response", {})
                                output_list = inner_response.get("output", [])

                                if output_list:
                                    first_output_item = output_list[0]
                                    print(first_output_item)
                                    content_list = first_output_item.get("content", [])

                                    if content_list:
                                        first_content_item = content_list[0]
                                        transcript = first_content_item.get("transcript")
                                        print(transcript)
                                        print('call transcripts', call_transcripts[session_id])

                                        if transcript:
                                            call_transcripts[session_id].append({
                                                "role": "assistant",
                                                "content": transcript
                                            })
                                        print(call_transcripts, "&&&&&&&&&&&&&&&&&&&&&&&&&")
                            except Exception as e:
                                logger.error(f"Error occurred in respose event {e}")
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
    finally:
        # Ensure we save call data even if there's an error
        if session_id:
            await store_call_data(session_id)


async def send_session_update(openai_ws, custom_prompt=None):
    """Send session update to OpenAI WebSocket."""
    session_update = {
        "type": "session.update",
        "session": {
            "input_audio_format": "g711_ulaw",
            "output_audio_format": "g711_ulaw",
            "voice": config.VOICE,
            "instructions": custom_prompt or SYSTEM_MESSAGE,
            "modalities": ["text", "audio"],
            "temperature": 0.8,
        }
    }
    print('Sending session update:', json.dumps(session_update))
    await openai_ws.send(json.dumps(session_update))
    logger.info("WebSocket handler completed")

@app.post("/send-payment-link")
async def send_payment_link(request: Request):
    """API endpoint to send payment link SMS to customer."""
    data = await request.json()
    phone_number = data.get("phone_number","+917011897710")
    amount = data.get("amount", "0")
    call_id = data.get("call_id")
    
    if not phone_number:
        return {"error": "Phone number is required"}
    
    # In production, you would integrate with an SMS provider
    # For now, we'll just log the request
    logger.info(f"Payment link SMS request: Phone: {phone_number}, Amount: {amount}, Call ID: {call_id}")
    
    # Twilio SMS :
    message = twilio_client.messages.create(
        body=f"Click here to make your payment of ₹{amount}: https://payment.example.com/{uuid.uuid4()}",
        from_=config.TWILIO_PHONE_NUMBER,
        to=phone_number
    )
    return {"success": True, "message": "Payment link sent successfully", "msg_sid": message.sid}


if __name__ == "__main__":
    # nosec B104: this host binding is intended in controlled environments
    # trunk-ignore(bandit/B104)
    uvicorn.run(app, host="0.0.0.0", port=config.PORT)