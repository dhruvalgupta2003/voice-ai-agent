import asyncio
from datetime import datetime
from enum import Enum
import json
import base64
import os
from typing import Dict, List, Optional,Any
import uuid
from urllib.parse import urlencode
from pydantic import BaseModel
import pywav

from fastapi import FastAPI, HTTPException, WebSocket, Request, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
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
from utils.transcribe_call_record import transcribe_user_call_record
from utils.rewrite_call_record import rewrite_call_record_with_user_conversation
from utils.date_formatter import format_due_date
# Load Config
config = Config()

# Validate Config
validate_config(config)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allowed all origins for now
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

# Global dictionary to track web clients connected to call updates
web_clients:Dict[str,Any] = {}

# ---------------------- broadcast to web clients --------------------------------
# Helper function to broadcast updates to web clients
async def broadcast_to_web_clients(call_id, event_data):
    """Send updates to all connected web clients for this call."""
    if call_id in web_clients:
        # Use asyncio.gather with exception handling to avoid crashes
        for client in web_clients[call_id]:
            logger.info(f"Sending message to all web clients{event_data}")
            try:
                await client.send_json(event_data)
            except Exception as e:
                logger.error(f"Error sending to web client: {e}")

# ------------ Save Raw Audio to .wav file ------------------------
async def save_raw_audio(session_id):
    try:
        filepath = f"recordings/{session_id}.wav"
        if os.path.exists(filepath):
            return True
        if session_id in raw_audio_buffers:
            os.makedirs("recordings", exist_ok=True)

            raw_data = b"".join(raw_audio_buffers[session_id])

            # Write µ-law encoded audio directly using pywav
            wave_write = pywav.WavWrite(filepath, 1, 8000, 8, 7)  # 1: mono, 8000Hz, 8bit, 7: µ-law
            wave_write.write(raw_data)
            wave_write.close()

            logger.info(f"Saved µ-law audio to {filepath}")
    except Exception as e:
        logger.error(f"Error saving raw audio: {e}")
        return False

# ---------- Store call Data ------------------
# Store call data (summary and tags)
async def store_call_data(session_id: str, call_id: str):
    try:
        logger.info(f"Generating call record for session id {session_id}")

        # First, ensure the audio is saved
        audio_saved = await save_raw_audio(session_id)

        if not audio_saved:
            logger.error(f"Failed to save audio for session {session_id}, cannot proceed with transcription")

        if session_id in call_transcripts and call_transcripts[session_id]:
            summary, tags = await generate_call_summary_and_tags(call_transcripts[session_id])

            # Create call record
            call_record = {
                "call_id": call_id,
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                "summary": summary,
                "tags": tags,
                "transcript": call_transcripts[session_id]
            }

            # In a production environment, you would store this in a database
            # For now, we'll save it to a JSON file
            os.makedirs("call_records", exist_ok=True)
            call_record_filepath = f"call_records/{call_id}.json"
            with open(call_record_filepath, "w") as f:
                json.dump(call_record, f, indent=2)

            # transcribe user_audio
            filepath = f"recordings/{session_id}.wav"
            if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
                logger.error(f"Audio file {filepath} does not exist or is empty, cannot transcribe")
            else:
                user_sentences = await transcribe_user_call_record(filepath)
                # rewrite call_records
                response = await rewrite_call_record_with_user_conversation(call_record_filepath, user_sentences)
                logger.info(f"Stored call data for session {session_id}")
                logger.info(f"Response : {response}")
            return call_record
    except Exception as e:
        logger.error(f"Failed to store call data for session {session_id}: {str(e)}")
        return None

@app.get("/")
async def index_page():
    return {"message": "Custom Voice Agent Media Stream Server is running!"}

@app.post("/make-call")
async def make_call(request: Request):
    """Make an outgoing call to the specified phone number."""
    data = await request.json()
    to_phone_number = data.get("to", "+917011897710")
    
    # Get loan account details - in production this would come from your CRM/database
    customer_name = data.get("customer_name", "Michael")
    account_number = data.get("account_number", "1234")
    loan_amount = data.get("loan_amount", "$10,000")
    due_date = data.get("due_date", "2025-05-25")
    
    parsed_due_date = format_due_date(due_date)
    due_date = datetime.strptime(due_date, '%Y-%m-%d').date()

    # Determine overdue or upcoming
    today = datetime.today().date()
    overdue_or_upcoming = "Overdue" if due_date < today else "Upcoming"
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
    custom_prompt = custom_prompt.replace("$XX,XXX", f"${loan_amount}")
    custom_prompt = custom_prompt.replace("[Date]", parsed_due_date)
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

@app.api_route("/outgoing-call", methods=["GET", "POST"])
async def handle_outgoing_call(request: Request):
    """Handle outgoing call and return TwiML response to connect to Media Stream."""
    # Get call_id from query parameters
    params = dict(request.query_params)
    call_id = params.get("call_id", "abcd")
    logger.info(f'outgoing call call id: {call_id}')
    
    response = VoiceResponse()
    response.say("Connecting your call to our loan collection assistant...")
    response.pause(length=1)
    
    connect = Connect()
    # The issue is here - WebSocket URL needs to be absolute, not relative
    if config.NGROK_URL is not None:
        websocket_url = f'{config.NGROK_URL.replace("http:", "ws:").replace("https:", "wss:")}/media-stream/{call_id}'
        connect.stream(url=websocket_url)
    else:
        # Handle the case when NGROK_URL is None
        logger.error("NGROK_URL is not configured")
    response.append(connect)
    return HTMLResponse(content=str(response), media_type="application/xml")

@app.post("/end-call/{call_sid}")
async def end_call(call_sid: str):
    try:
        # Try to fetch the call first (optional but good for checking)
        twilio_client.calls(call_sid).update(status="completed")
        return {"message": f"Call {call_sid} ended successfully", "status": "completed"}
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Failed to end call: {str(e)}") from e

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

@app.websocket("/call-updates/{call_id}")
async def call_updates(websocket: WebSocket, call_id: str):
    """Websocket endpoint to receive call updates."""
    logger.info(f'Connected to call updates for call_id: {call_id}')
    await websocket.accept()
    # Store this connection in the global dictionary
    if call_id not in web_clients:
        web_clients[call_id] = []
    web_clients[call_id].append(websocket)

    logger.info(f"Web client connected to call updates for call_id: {call_id}")
    await websocket.send_json({"event_type": "connected", "message": "Connected to call updates"})

    try:
        # Keep connection alive and handle incoming messages if needed
        while True:
            data = await websocket.receive_text()
            logger.info(f"Recieved Data for client: {data}")
            # Process any messages from web client if needed
    except WebSocketDisconnect:
        # Remove this websocket from tracking
        if call_id in web_clients and websocket in web_clients[call_id]:
            web_clients[call_id].remove(websocket)
            if not web_clients[call_id]:
                del web_clients[call_id]
        logger.info(f"Web client disconnected from call updates for call_id: {call_id}")
        await websocket.send_json({"event_type": "disconnected", "message": "call disconnected"})


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
            openai_session_id = None
            
            # Flag to track if the connection is still valid
            connection_active = True

            async def receive_from_twilio():
                """Receive audio data from Twilio and send it to the OpenAI Realtime API."""
                nonlocal stream_sid, connection_active
                try:
                    async for message in websocket.iter_text():
                        data = json.loads(message)
                        if data['event'] == 'media' and connection_active:
                            base64_audio = data['media']['payload']
                            # Decode and save to buffer
                            raw_audio_buffers[session_id].append(base64.b64decode(base64_audio))

                            # Forward to OpenAI
                            audio_append = {
                                "type": "input_audio_buffer.append",
                                "audio": base64_audio
                            }
                            try:
                                await openai_ws.send(json.dumps(audio_append))
                            except Exception as e:
                                logger.error(f"Failed to send to OpenAI: {e}")
                                connection_active = False
                        elif data['event'] == 'start':
                            stream_sid = data['start']['streamSid']
                            logger.info(f"Incoming stream has started {stream_sid}")
                        elif data['event'] == 'stop':
                            logger.info('Twilio Stopped....')
                            if call_id and session_id:
                                logger.info(f"saving audio and call_transcripts for session : {session_id}")
                                await save_raw_audio(session_id)
                                await store_call_data(session_id, call_id)
                            # Send close notification to all web clients for this call
                            if call_id in web_clients:
                                for client in web_clients[call_id]:
                                    try:
                                        await client.send_json({
                                            "event_type": "connection_closed",
                                            "message": "Call ended normally"
                                        })
                                    except Exception as e:
                                        logger.error(f"Error sending close notification: {e}")
                except WebSocketDisconnect: 
                    logger.info("Client disconnected.")
                    connection_active = False
                    # Call ended, generate summary and store data
                    await broadcast_to_web_clients(call_id, {
                                                    "event_type": "connection closed",
                                                    "message": "Call ended"
                                                })
                    if call_id and session_id:
                        await store_call_data(session_id, call_id)
                    try:
                        await openai_ws.close()
                    except Exception as e:
                        logger.error(f"error in twilio websocket {e}")

            async def send_to_twilio():
                """Receive events from the OpenAI Realtime API, send audio back to Twilio."""
                nonlocal stream_sid, openai_session_id
                try:
                    async for openai_message in openai_ws:
                        response = json.loads(openai_message)
                        if response['type'] in config.LOG_EVENT_TYPES:
                            logger.info(f"Received event: {response['type']} {response}")
                        
                        # store session ID
                        if response['type'] == 'session.created':
                            openai_session_id = response['session']['id']
                        
                        if response['type'] == 'session.updated':
                            logger.info("Session updated successfully:", response)
                        
                        # Handle tool calls
                        if response['type'] == 'response.function_call_arguments.done':
                            logger.info(f'Tool calling response {response}')
                            tool_name = response["name"]
                            function_call_id = response["call_id"]  # Renamed to avoid conflict
                            logger.info(f"Tool call function id : {function_call_id}")
                            raw_args = response.get("arguments", "{}")

                            try:
                                arguments = json.loads(raw_args)
                            except Exception as e:
                                logger.error(f"Error decoding tool arguments: {e}")
                                arguments = {}

                            # Optional: include call_id in parameters for logging or audit
                            arguments["call_id"] = call_id  # This uses the outer call_id parameter
                            logger.info(f"Calling execute_tool with {tool_name}, args: {arguments}")
                            result = await execute_tool(tool_name, arguments)
                            logger.info(f'Result of tool calling{json.dumps(result, indent=2)}')
                        # Handle transcripts for summary
                        if response['type'] == 'conversation.item.created':
                            logger.info(f"conversation.item.created event: {response}")
                            # Store conversation transcript
                            if 'item' in response and response['item']['type'] == 'function_call':
                                logger.info(f"Function call received: {json.dumps(response, indent=2)}")

                            if 'item' in response and 'content' in response['item']:
                                for content_block in response['item']['content']:
                                    if content_block.get("type") == "input_audio":
                                        transcript = content_block.get("transcript")
                                        if transcript:
                                            call_transcripts[session_id].append({
                                                'role': response['item']['role'],
                                                'content': transcript
                                            })
                                            if call_id:  # This references the outer call_id parameter
                                                await broadcast_to_web_clients(call_id, {
                                                    "event_type": "transcript",
                                                    "role": "user",
                                                    "content": transcript
                                                })

                        if response.get("type") == "response.done":
                            try:
                                inner_response = response.get("response", {})
                                output_list = inner_response.get("output", [])

                                if output_list:
                                    first_output_item = output_list[0]
                                    content_list = first_output_item.get("content", [])

                                    if content_list:
                                        first_content_item = content_list[0]
                                        transcript = first_content_item.get("transcript")

                                        if transcript:
                                            call_transcripts[session_id].append({
                                                "role": "assistant",
                                                "content": transcript
                                            })
                                            # Broadcast to web clients - Use the outer call_id parameter
                                            if call_id:  # This references the outer call_id parameter
                                                await broadcast_to_web_clients(call_id, {
                                                    "event_type": "transcript",
                                                    "role": "assistant",
                                                    "content": transcript
                                                })
                            except Exception as e:
                                logger.error(f"Error occurred in response event: {e}")
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
                                logger.error(f"Error processing audio data: {e}")
                        if response['type'] == 'conversation.item.created':
                            logger.info(f"conversation.item.created event: {response}")
                except Exception as e:
                    logger.error(f"Error in send_to_twilio: {e}")

            await asyncio.gather(receive_from_twilio(), send_to_twilio())
    except Exception as e:
        logger.error(f"Error Occurred {e}")
    finally:
        # Ensure we save call data even if there's an error
        if call_id and session_id:
            await store_call_data(session_id, call_id)

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
            "tools": [
                {
                    "type": "function",
                    "name": "send_payment_link",
                    "description": "Send a payment link via SMS to a customer using their phone number, amount, and optional call ID.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "phone_number": {
                                "type": "string",
                                "description": "Recipient's phone number is +917011897710"
                            },
                            "amount": {
                                "type": "number",
                                "description": "Amount to be paid by the customer (due amount) get it from custom prompt"
                            },
                            "call_id": {
                                "type": "string",
                                "description": "Optional call session ID related to the payment",
                                "nullable": True
                            }
                        },
                        "required": ["phone_number", "amount"]
                    }
                },
                {
                    "type": "function",
                    "name": "get_payment_status",
                    "description": "Retrieve the payment status for a given call ID including status and amount details.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                        "call_id": {
                            "type": "string",
                            "description": "The unique identifier of the call whose payment status is being requested."
                        }
                        },
                        "required": ["call_id"]
                    }
                }
            ],
            "tool_choice": "auto",
        }
    }
    logger.info(f'Sending session update: {json.dumps(session_update, indent=2)}')
    await openai_ws.send(json.dumps(session_update))
    logger.info("WebSocket handler completed")

async def execute_tool(tool_name, parameters):
    """
    Execute the requested tool based on the parameters provided.

    Supported tools:
    - send_payment_link: Sends a payment link to the user's phone number.
    - get_payment_status: Retrieves the payment status for a specific call ID.

    Args:
        tool_name (str): The name of the tool to execute.
        parameters (dict): A dictionary of parameters required by the tool.

    Returns:
        dict: A dictionary with the execution result.
    """
    logger.info(f"Executing tool call: {tool_name} with params: {parameters}")

    if tool_name == "send_payment_link":
        phone_number = parameters.get("phone_number")
        amount = parameters.get("amount")
        call_id = parameters.get("call_id")

        if not phone_number or not amount or not call_id:
            return {
                "success": False,
                "message": "Missing parameters for sending payment link. Required: phone_number, amount, call_id"
            }

        logger.info(f"Sending payment link: ${amount} to {phone_number}, call_id: {call_id}")
        result = await send_payment_link(phone_number, amount, call_id)
        logger.info(f"send_payment_link result: {result}")

        return {
            "success": True,
            "message": f"Payment link for ${amount} sent to {phone_number}",
            "details": result
        }

    elif tool_name == "get_payment_status":
        call_id = parameters.get("call_id")

        if not call_id:
            return {
                "success": False,
                "message": "Missing parameter: call_id is required to fetch payment status."
            }

        try:
            payment_status = await get_payment_status(call_id)
            logger.info(f"Payment status for call_id {call_id}: {payment_status}")
            return {
                "success": True,
                "message": "Payment status retrieved successfully.",
                "status": payment_status
            }
        except HTTPException as e:
            logger.error(f"Error retrieving payment status: {e.detail}")
            return {
                "success": False,
                "message": e.detail
            }

    else:
        logger.warning(f"Unknown tool requested: {tool_name}")
        return {
            "success": False,
            "message": f"Unknown tool: {tool_name}"
        }

def generate_mock_payment_link(phone_number: str, amount: float, call_id: str = 'xyz') -> str:
    """
    Generate a mock payment link hosted on the backend server.

    Parameters:
    - phone_number (str): The recipient's phone number.
    - amount (float): Payment amount.
    - call_id (str): Optional call ID for session tracking.

    Returns:
    - str: Fully constructed mock payment URL.
    """
    base_url = f"{config.NGROK_URL}/demo-pay"
    params = {
        "phone": phone_number,
        "amount": amount,
    }
    if call_id:
        params["call_id"] = call_id

    return f"{base_url}?{urlencode(params)}"

class PaymentLinkRequest(BaseModel):
    phone_number: str
    amount: float
    call_id: Optional[str] = None


async def send_payment_link(phone_number: str, amount: float, call_id: Optional[str] = None):
    # Step 1: Generate mock payment link
    payment_link = generate_mock_payment_link(
        phone_number=phone_number,
        amount=amount,
        call_id=call_id or 'xyz'
    )

    # Step 2: Send SMS using Twilio
    message_body = f"Make Payment of ${amount:.2f}: {payment_link}",
    message = twilio_client.messages.create(
        body=message_body,
        from_=config.TWILIO_PHONE_NUMBER,
        to=phone_number
    )
    logger.info(f"Message Body: {message_body}")

    logger.info(f"Payment link sent. Message SID: {message.sid}, payment link : {payment_link}")
    return {
        "success": True,
        "message": "Payment link sent successfully",
        "msg_sid": message.sid,
        "link": payment_link
    }

async def get_payment_status(call_id: str) -> Dict[str, Any]:
    """
    Retrieve the payment status for a given call ID.

    This function looks for a JSON file in the 'payment_status' directory,
    named using the provided call ID. If found and valid, it returns the
    payment status data.

    Args:
        call_id (str): Unique identifier for the call whose payment status is being retrieved.

    Returns:
        Dict[str, Any]: A dictionary containing payment status details, e.g.:
            {
                "status": True,
                "amount": "$499"
            }

    Raises:
        HTTPException:
            - 400: If the call_id is missing or not a string.
            - 404: If the file does not exist.
            - 500: If the file is unreadable or contains invalid/corrupt data.
    """
    if not isinstance(call_id, str) or not call_id.strip():
        raise HTTPException(status_code=400, detail="Invalid or missing call ID.")

    file_path = os.path.join("payment_status", f"{call_id}.json")

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Payment status not found for the provided call ID.")

    try:
        with open(file_path, "r") as f:
            payment_status = json.load(f)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail="Payment status file is corrupted or invalid.") from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error while reading payment status: {str(e)}") from e

    if not isinstance(payment_status, dict):
        raise HTTPException(status_code=500, detail="Unexpected format in payment status file.")

    return payment_status

@app.post("/send-payment-link")
async def send_payment_link_api(request: PaymentLinkRequest):
    """
    Sends a payment link via SMS to the specified phone number.

    Parameters:
    - phone_number (str): The recipient's phone number (in E.164 format, e.g. +9170xxxxxxx).
    - amount (float): The amount to be paid.
    - call_id (Optional[str]): The call session ID related to the payment (if applicable).

    Returns:
    - dict: Contains success status, message, payment link, and Twilio message SID if successful.

    Raises:
    - HTTPException: If SMS sending fails due to an external error or invalid parameters.

    Example:
    >>> send_payment_link({
            "phone_number": "+917011897710",
            "amount": 349.99,
            "call_id": "abc123"
        })
    """
    try:
        if not request.phone_number:
            raise HTTPException(status_code=400, detail="Phone number is required")

        logger.info(f"Sending payment link to: {request.phone_number}, Amount: ${request.amount:.2f}, Call ID: {request.call_id}")

        # Step 1: Generate mock payment link
        payment_link = generate_mock_payment_link(
            phone_number=request.phone_number,
            amount=request.amount,
            call_id=request.call_id or 'xyz'
        )


        # Step 3: Send SMS using Twilio
        message = twilio_client.messages.create(
            body=f"Click here to make your payment of ${request.amount:.2f}: {payment_link}",
            from_=config.TWILIO_PHONE_NUMBER,
            to=request.phone_number
        )

        logger.info(f"Payment link sent. Message SID: {message.sid}, payment link : {payment_link}")
        return {
            "success": True,
            "message": "Payment link sent successfully",
            "msg_sid": message.sid,
            "link": payment_link
        }

    except Exception as e:
        logger.error(f"Failed to send payment link: {e}")
        # trunk-ignore(ruff/B904)
        raise HTTPException(status_code=500, detail="Failed to send payment link")

@app.post("/payment-status")
async def update_payment_status(data: dict):
    """
    Mock endpoint to update payment status when user clicks 'Pay' on demo page.
    """
    import os
    import json
    from fastapi import HTTPException

    phone = data.get("phone")
    call_id = data.get("call_id")
    amount = data.get("amount")

    if not (phone and call_id and amount):
        raise HTTPException(status_code=400, detail="Missing phone, call_id, or amount")

    file_path = f'payment_status/{call_id}.json'
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Check existing file
    payment_status = {'status': None, 'amount': None}
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            try:
                payment_status = json.load(f)
            except json.JSONDecodeError:
                payment_status = {'status': None, 'amount': None}

    # If payment has not been recorded or file was invalid, update it
    if not payment_status['status']:
        with open(file_path, 'w') as f:
            json.dump({
                "status": True,
                "amount": f"${amount}"
            }, f)

    logger.info(f"Mock payment received from {phone} for call ID: {call_id} — Amount: {amount}")

    return {"success": True, "message": "Payment status updated"}

@app.get("/payment-status/{call_id}", response_model=Dict[str, Any])
async def get_payment_status_api(call_id: str):
    """
    API Endpoint: Retrieve the payment status for a given call ID.
    """
    if not isinstance(call_id, str) or not call_id.strip():
        raise HTTPException(status_code=400, detail="Invalid or missing call ID.")

    file_path = os.path.join("payment_status", f"{call_id}.json")

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Payment status not found for the provided call ID.")

    try:
        with open(file_path, "r") as f:
            payment_status = json.load(f)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail="Payment status file is corrupted or invalid.") from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error while reading payment status: {str(e)}") from e

    if not isinstance(payment_status, dict):
        raise HTTPException(status_code=500, detail="Unexpected format in payment status file.")

    return JSONResponse(content=payment_status)

@app.get("/demo-pay", response_class=HTMLResponse)
async def demo_pay(phone: str, amount: float, call_id: Optional[str] = None):
    return f"""
    <html>
    <head>
        <title>Mock Payment</title>
        <style>
            body {{ font-family: 'Segoe UI', sans-serif; text-align: center; padding: 40px; background: #f9f9f9; }}
            .card {{ background: white; padding: 20px; margin: auto; border-radius: 10px; box-shadow: 0 0 20px rgba(0,0,0,0.1); max-width: 400px; }}
            .btn {{ background: #007bff; color: white; padding: 10px 20px; border-radius: 5px; border: none; font-size: 16px; cursor: pointer; }}
            .btn:hover {{ background: #0069d9; }}
            .success {{ color: #28a745; margin-top: 20px; display: none; }}
            .error {{ color: #dc3545; margin-top: 20px; display: none; }}
        </style>
    </head>
    <body>
        <div class="card">
            <h2>Pay Now</h2>
            <p><strong>Phone:</strong> {phone}</p>
            <p><strong>Amount:</strong> ${amount:.2f}</p>
            <p><strong>Call ID:</strong> {call_id or 'N/A'}</p>
            <button class="btn" onclick="handleMockPayment()">Pay ${amount:.2f}</button>
            <p id="success-message" class="success">Payment successful!</p>
            <p id="error-message" class="error">Payment update failed. Please try again.</p>
        </div>
        <script>
            async function handleMockPayment() {{
                alert('This is a mock payment — no money was transferred!');
                const payload = {{
                    phone: "{phone}",
                    call_id: "{call_id or ''}",
                    status: "paid",
                    amount: {amount}
                }};
                try {{
                    document.querySelector('.btn').disabled = true;
                    document.querySelector('.btn').innerHTML = 'Processing...';
                    const res = await fetch("{config.NGROK_URL}/payment-status", {{
                        method: "POST",
                        headers: {{ "Content-Type": "application/json" }},
                        body: JSON.stringify(payload)
                    }});
                    const result = await res.json();
                    console.log("Payment status update response:", result);
                    document.getElementById('success-message').style.display = 'block';
                    document.querySelector('.btn').innerHTML = 'Paid';
                    document.querySelector('.btn').style.background = '#28a745';
                    // Notify parent window if this is in an iframe
                    if (window.opener) {{
                        window.opener.postMessage({{ type: 'PAYMENT_COMPLETE', status: 'success' }}, '*');
                    }}
                    // Close window after delay if opened as popup
                    setTimeout(() => {{
                        if (window.opener) {{
                            window.close();
                        }}
                    }}, 3000);

                }} catch (error) {{
                    console.error("Failed to update payment status:", error);
                    document.getElementById('error-message').style.display = 'block';
                    document.querySelector('.btn').disabled = false;
                    document.querySelector('.btn').innerHTML = 'Try Again';
                }}
            }}
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    # nosec B104: this host binding is intended in controlled environments
    # trunk-ignore(bandit/B104)
    uvicorn.run(app, host="0.0.0.0", port=config.PORT)