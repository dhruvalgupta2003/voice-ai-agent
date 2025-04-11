from fastapi import FastAPI, WebSocket, Request, Response, WebSocketDisconnect, Query
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from twilio.twiml.voice_response import VoiceResponse, Connect
from twilio.rest import Client
import websockets
from prompts import Loan_Agent_System_Prompt
from groq import Groq
from config import Config
import assemblyai as aai
import json
import uuid
import logging


# Initialize FastAPI app
app = FastAPI(title="Twilio Voice AI Assistant")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load config
config = Config()
logger.info('...LOgiging config')
# Initialize twilio client
twilio_client = Client(config.TWILIO_ACCOUNT_SID, config.TWILIO_AUTH_TOKEN)

# Initialize groq Client
groq_client = Groq(api_key=config.GROQ_API_KEY)

ngrok_url = 'https://4cab-139-5-18-82.ngrok-free.app'

# Initial messages for conversation
initial_messages = [
    {"role": "system", "content": Loan_Agent_System_Prompt}
]

# Store active calls with unique session IDs
active_calls = {}

class PhoneCallRequest(BaseModel):
    phone_number: str

# Handle incoming voice call
@app.post("/voice")
async def voice(request: Request):
    """Handle incoming voice calls"""
    form_data = await request.form()
    call_sid = form_data.get('CallSid')
    
    response = VoiceResponse()
    
    # Generate a unique session ID for this call
    session_id = str(uuid.uuid4())
    active_calls[call_sid] = {
        'session_id': session_id,
        'messages': initial_messages.copy()  # Copy the initial messages
    }
    
    # Connect to our WebSocket stream endpoint
    response.connect().stream(url=f'{ngrok_url}/stream?session_id={session_id}')
    
    return Response(content=str(response), media_type="application/xml")

@app.post("/make-call")
async def make_call(request: Request):
    """Make an outgoing call to the specified phone number."""
    data = await request.json()
    phone_number = data.get('phone_number')
    
    if not phone_number:
        return {"error": "Phone number is required"}
    
    client = Client(config.TWILIO_ACCOUNT_SID, config.TWILIO_AUTH_TOKEN)
    call = client.calls.create(
        url = f"{ngrok_url}/outgoing-call",
        to = phone_number,
        from_ = config.TWILIO_PHONE_NUMBER,
    )
    return {"success": True, "sid": call.sid}

@app.api_route("/outgoing-call", methods=["GET", "POST"])
async def handle_outgoing_call(request: Request):
    """Handle outgoing calls"""
    response = VoiceResponse()
    response.say("Please wait we are connecting you to Ava (your personalized voice assistant)")
    response.pause(length=1)
    response.say("Please speak now...")
    connect = Connect()
    connect.stream(url=f'wss://{request.url.hostname}/media-stream')
    response.append(connect)
    return HTMLResponse(content=str(response), media_type="application/xml")

# Handle outbound calls
@app.post("/call", response_class=JSONResponse)
async def call(request: PhoneCallRequest):
    """Initiate an outbound call to the specified phone number with media streaming"""
    try:
        phone_number = request.phone_number
        
        # Generate a unique session ID for this call
        session_id = str(uuid.uuid4())
        
        # Create a call using the Twilio client
        call = twilio_client.calls.create(
            to=phone_number,
            from_=config.TWILIO_PHONE_NUMBER,
            url=f"{ngrok_url}/voice?session_id={session_id}"
        )
        
        # Store this call in active calls
        active_calls[call.sid] = {
            'session_id': session_id,
            'messages': initial_messages.copy()  # Copy the initial messages
        }
        
        return {"success": True, "call_sid": call.sid}
    except Exception as e:
        logger.error(f"Error initiating call: {e}")
        return JSONResponse(status_code=500, content={"error": f"Error initiating call: {str(e)}"})

@app.websocket("/media-stream")
async def handle_media_stream(websocket: WebSocket)
    """Handle WebSocket connections between Twilio and OpenAI."""
    logger.info("Client connected")
    
    await websocket.accept()
    async with websockets.connect(
        'wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01',
        extra_headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "OpenAI-Beta": "realtime=v1"
        }
    ) as openai_ws:
        await send_session_update(openai_ws)
        stream_sid = None
        session_id = None

        async def receive_from_twilio():
            """Receive audio data from Twilio and send it to the OpenAI Realtime API."""
            nonlocal stream_sid
            try:
                async for message in websocket.iter_text():
                    data = json.loads(message)
                    if data['event'] == 'media' and openai_ws.open:
                        audio_append = {
                            "type": "input_audio_buffer.append",
                            "audio": data['media']['payload']
                        }
                        await openai_ws.send(json.dumps(audio_append))
                    elif data['event'] == 'start':
                        stream_sid = data['start']['streamSid']
                        print(f"Incoming stream has started {stream_sid}")
            except WebSocketDisconnect:
                print("Client disconnected.")
                if openai_ws.open:
                    await openai_ws.close()

        async def send_to_twilio():
            """Receive events from the OpenAI Realtime API, send audio back to Twilio."""
            nonlocal stream_sid, session_id
            try:
                async for openai_message in openai_ws:
                    response = json.loads(openai_message)
                    if response['type'] in LOG_EVENT_TYPES:
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

async def send_session_update(openai_ws):
    """Send session update to OpenAI WebSocket."""
    session_update = {
        "type": "session.update",
        "session": {
            "input_audio_format": "g711_ulaw",
            "output_audio_format": "g711_ulaw",
            "voice": VOICE,
            "instructions": SYSTEM_MESSAGE,
            "modalities": ["text", "audio"],
            "temperature": 0.8,
        }
    }
    print('Sending session update:', json.dumps(session_update))
    await openai_ws.send(json.dumps(session_update))

# if __name__ == "__main__":
#     import uvicorn
#     to_phone_number = input("Please enter the phone number to call: ")
#     client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
#     try:
#         call = client.calls.create(
#             url=f"{NGROK_URL}/outgoing-call",
#             to=to_phone_number,
#             from_=TWILIO_PHONE_NUMBER
#         )
#         print(f"Call initiated with SID: {call.sid}")
#     except Exception as e:
#         print(f"Error initiating call: {e}")
#     uvicorn.run(app, host="0.0.0.0", port=PORT)
    
# WebSocket stream endpoint
@app.websocket("/stream")
async def stream_endpoint(websocket: WebSocket, session_id: str = Query(...)):
    """Handle media streams from Twilio as a WebSocket connection"""
    await websocket.accept()
    logger.info(f"WebSocket connection established for session {session_id}")
    
    # Find the call data using session_id
    call_data = None
    for sid, data in active_calls.items():
        logger.info(f'Session ID : f{sid}')
        if data['session_id'] == session_id:
            call_data = data
            break
    
    if not call_data:
        await websocket.close(code=1008, reason="Session not found")
        return
    
    try:
        # Initial greeting from the AI
        initial_message = {"role": "assistant", "content": "Hello, I'm your AI loan assistant. How can I help you today?"}
        await websocket.send_text(json.dumps({
            "event": "media",
            "streamSid": "initial",  # This will be replaced by the actual StreamSid
            "response": initial_message["content"]
        }))
        
        # Add the initial message to the conversation history
        call_data['messages'].append(initial_message)
        
        while True:
            # Receive audio data from Twilio
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Handle the different message types
            if message["event"] == "media":
                # This is audio data - process with speech-to-text
                audio_data = message["media"]["payload"]
                
                # Convert audio to text (you would implement this function)
                user_text = convert_audio_to_text(audio_data)
                logger.info(f"Transcribed text: {user_text}")
                
                # Add user message to history
                call_data['messages'].append({"role": "user", "content": user_text})
                
                # Generate AI response using Groq
                ai_response = await generate_ai_response(call_data['messages'])
                
                # Add AI response to history
                call_data['messages'].append({"role": "assistant", "content": ai_response})
                logger.info(f"AI response: {ai_response}")
                
                # Send AI response back to Twilio for conversion to speech
                await websocket.send_text(json.dumps({
                    "event": "media",
                    "streamSid": message.get("streamSid", "unknown"),
                    "response": ai_response
                }))
            
            elif message["event"] == "stop":
                # Call is ending, clean up resources
                for sid, data in active_calls.items():
                    if data['session_id'] == session_id:
                        del active_calls[sid]
                        logger.info(f"Removed session {session_id} from active calls")
                        break
                break
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {session_id}")
        # Clean up if needed
        for sid, data in list(active_calls.items()):
            if data['session_id'] == session_id:
                del active_calls[sid]
                break
    except Exception as e:
        logger.error(f"Error in WebSocket communication: {e}")
        # Clean up if needed
        for sid, data in list(active_calls.items()):
            if data['session_id'] == session_id:
                del active_calls[sid]
                break



def convert_audio_to_text(audio_data):
    """Transcribe audio using AssemblyAI"""
    try:
        # If audio_data is a bytearray, convert to bytes
        if isinstance(audio_data, bytearray):
            audio_data = bytes(audio_data)
        
        # Save to a temporary file if needed by AssemblyAI
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            temp_file.write(audio_data)
            temp_file_path = temp_file.name

        # Transcribe the saved file
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(temp_file_path)
        
        # Clean up temporary file
        import os
        os.unlink(temp_file_path)

        logger.info(f"Transcribed Text: {transcript.text}")
        
        return transcript.text or "No transcription available"
    except Exception as e:
        logger.error(f"Error transcribing audio: {e}")
        return None

def generate_ai_response(message_history):
    """Generate an AI response using Groq"""
    try:
        response = groq_client.chat.completions.create(
            model="llama3-70b-8192",
            messages=message_history
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating AI response: {e}")
        return "I'm sorry, I'm having trouble processing your request right now."

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)