
import asyncio
import os
import json
import base64
import logging
import requests
from fastapi import FastAPI, HTTPException, WebSocket, Request
from fastapi.responses import HTMLResponse
from fastapi.websockets import WebSocketDisconnect
from twilio.twiml.voice_response import VoiceResponse, Connect, Say, Stream, Gather
from dotenv import load_dotenv
import assemblyai as aai
import groq
import uvicorn
import boto3

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Configuration
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
ASSEMBLYAI_API_KEY = os.getenv('ASSEMBLYAI_API_KEY')
TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')

SYSTEM_MESSAGE = (
    "You are a helpful and bubbly AI assistant who loves to chat about "
    "anything the user is interested in and is prepared to offer them facts. "
    "You have a penchant for dad jokes, owl jokes, and rickrolling – subtly. "
    "Always stay positive, but work in a joke when appropriate.")

app = FastAPI()
groq_client = groq.Groq(api_key=GROQ_API_KEY)
aai.settings.api_key = ASSEMBLYAI_API_KEY

# Validate environment variables
if not all([GROQ_API_KEY, ASSEMBLYAI_API_KEY, TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN]):
    raise ValueError("Missing required environment variables")

@app.api_route("/", methods=["GET", "POST"])
async def index_page():
    return "<h1>Server is up and running.</h1>"

@app.api_route("/incoming-call", methods=["GET", "POST"])
async def handle_incoming_call(request: Request):
    """Handle incoming call and return TwiML response"""
    try:
        response = VoiceResponse()
        response.say("Please wait while we connect your call to the A.I. voice assistant.")
        response.pause(length=1)
        response.say("OK, you can start talking!")
        
        host = request.url.hostname
        # host = 'd58d-2405-201-4046-c003-715c-b8da-658a-4e25.ngrok-free.app'
        connect = Connect()
        connect.stream(url=f'wss://{host}/media-stream')
        response.append(connect)
        
        return HTMLResponse(content=str(response), media_type="application/xml")
    except Exception as e:
        logger.error(f"Error handling incoming call: {e}")
        raise

# @app.post("/make-call")
# async def make_call(phone_number: str):
#     """Initiate an outbound call to the specified phone number"""
#     try:
#         from twilio.rest import Client
        
#         # Initialize Twilio client
#         client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        
#         # Make the call
#         call = client.calls.create(
#             to=phone_number,
#             from_=os.getenv('TWILIO_PHONE_NUMBER'),
#             url=f"https://d58d-2405-201-4046-c003-715c-b8da-658a-4e25.ngrok-free.app/incoming-call"
#         )
        
#         return {"status": "success", "call_sid": call.sid}
#     except Exception as e:
#         logger.error(f"Error making outbound call: {e}")
#         return {"status": "error", "message": str(e)}

@app.post("/make-call")
async def make_call(phone_number: str = "+917011897710"):
    """Initiate an outbound call to the specified phone number with media streaming"""
    if not phone_number:
        try:
            body = await requests.request.json()
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
        # host = os.getenv('WEBHOOK_HOST', 'localhost:8080')  # Use environment variable for flexibility
        host= "4cab-139-5-18-82.ngrok-free.app"
        connect.stream(url=f'wss://{host}/media-stream')
        twiml.append(connect)
        
        # Initialize Twilio client
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        
        # Make the call with dynamically generated TwiML
        call = client.calls.create(
            to=phone_number,
            from_=os.getenv('TWILIO_PHONE'),
            twiml=str(twiml)
        )
        print(os.getenv('TWILIO_PHONE'))
        
        return {"status": "success", "call_sid": call.sid}
    except Exception as e:
        logger.error(f"Error making outbound call: {e}")
        return {"status": "error", "message": str(e)}

async def transcribe_audio(audio_data):
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

async def get_llm_response(text):
    """Get response from Groq LLM"""
    try:
        logger.info(f"LLM REQUEST TEXT: {text}")
        completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": text}
            ],
            model="llama3-70b-8192",
            temperature=0.7,
        )
        logger.info(f'LLM RESPONSE: {completion.choices[0].message.content}')
        return completion.choices[0].message.content
    except Exception as e:
        logger.error(f"Error getting LLM response: {e}")
        return None

async def text_to_speech(text):
    """Convert text to speech using AWS Polly"""
    try:
        # Ensure output directory exists
        os.makedirs('output_dir', exist_ok=True)

        polly_client = boto3.Session(
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_REGION", "us-east-1")).client("polly")
        response = polly_client.synthesize_speech(
                Text=text,
                OutputFormat='mp3',
                VoiceId='Ruth',  # You can change the voice as needed
                Engine="neural"
        )
        if "AudioStream" in response:
            # Base64 encode for WebSocket
            # Read the audio stream
            audio_stream = response['AudioStream'].read()
            
            base64_audio = base64.b64encode(audio_stream).decode('utf-8')
            
            # Save the MP3 file
            output_audio_path = os.path.join('output_dir', 'output_audio.mp3')
            with open(output_audio_path, 'wb') as f:
                f.write(audio_stream)
            
            print(f"Output audio saved to {output_audio_path}")
            
            return base64_audio
        return None
    except Exception as e:
        logger.error(f"Error converting text to speech: {e}")
        return None

# @app.websocket("/media-stream")
# async def handle_media_stream(websocket: WebSocket):
#     """Handle WebSocket connections for the media stream"""
#     logger.info("Client connected")
#     has_seen_media = False
#     message_count = 0
#     await websocket.accept()
#     stream_sid = None
#     audio_buffer = bytearray()

#     try:
#         while True:
#             message = await websocket.receive_text()
#             data = json.loads(message)
#             if data['event'] == 'connected':
#                 print("connected message is received....")
#             if data['event'] == 'media':
#                 print('media-event is running....')
#                 # Accumulate audio data
#                 payload = data['media']['payload']
#                 audio_chunk = base64.b64decode(payload)
#                 print(f"received {len(audio_chunk)} bytes for processing...")
#                 audio_buffer.extend(audio_chunk)
#                 has_seen_media = True
#             elif data['event'] == 'start':
#                 print('start-event is running....')
#                 stream_sid = data['start']['streamSid']
#                 logger.info(f"Stream started: {stream_sid}")
#             elif data['event'] == 'stop' or len(audio_buffer) >= 32000:  # Process in chunks
#                 print('end event is running ...')
#                 # Process accumulated audio
#                 if audio_buffer:
#                     # Transcribe audio
#                     transcript = await transcribe_audio(audio_buffer)
#                     if transcript:
#                         # Get LLM response
#                         llm_response = await get_llm_response(transcript)
#                         if llm_response:
#                             # Convert response to speech
#                             audio_payload = await text_to_speech(llm_response)
#                             if audio_payload:
#                                     await websocket.send_json({
#                                         "event": "media",
#                                         "streamSid": stream_sid,
#                                         "media": {
#                                             "payload": audio_payload
#                                         }
#                                     })
#                     # Clear buffer after processing
#                     audio_buffer = bytearray()
#             message_count +=1

#     except WebSocketDisconnect:
#         logger.info("Client disconnected")
#     except Exception as e:
#         logger.error(f"Error in media stream handling: {e}")
#     finally:
#         try:
#             await websocket.close()
#         except Exception as e:
#             print(e)
@app.websocket("/media-stream")
async def handle_media_stream(websocket: WebSocket):
    """Handle WebSocket connections for the media stream"""
    logger.info("Client connected")
    await websocket.accept()
    
    stream_sid = None
    audio_buffer = bytearray()
    buffer_threshold = 20000  # Buffer size for processing
    is_processing = False

    # Define the helper function within the proper scope
    def set_processing_done(_):
        nonlocal is_processing
        is_processing = False

    try:
        while True:
            message = await websocket.receive_text()
            data = json.loads(message)
            
            if data['event'] == 'connected':
                logger.info("Websocket connection established")
            elif data['event'] == 'start':
                stream_sid = data['start']['streamSid']
                logger.info(f"Stream started: {stream_sid}")
            elif data['event'] == 'media':
                payload = data['media']['payload']
                audio_chunk = base64.b64decode(payload)
                audio_buffer.extend(audio_chunk)
                
                # Log the current buffer size
                logger.info(f"Buffer size: {len(audio_buffer)}")
                # Process in real-time when buffer threshold is reached and not already processing
                if len(audio_buffer) >= buffer_threshold and not is_processing and stream_sid:
                    logger.info(f"Processing audio buffer of size {len(audio_buffer)}")
                    # Create a copy of the current buffer for processing
                    current_buffer = bytes(audio_buffer)
                    # Clear buffer immediately
                    audio_buffer = bytearray()
                    # Start async processing without waiting for it to complete
                    is_processing = True
                    processing_task = asyncio.create_task(
                        process_and_respond(websocket, stream_sid, current_buffer)
                    )
                    # Add callback to reset processing flag when done - properly scoped now
                    processing_task.add_done_callback(set_processing_done)
            elif data['event'] == 'stop':
                logger.info("Stream stopped")
                # Process any remaining audio
                if audio_buffer and stream_sid:
                    await process_and_respond(websocket, stream_sid, bytes(audio_buffer))
                break
    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"Error in media stream handling: {e}", exc_info=True)
    finally:
        try:
            await websocket.close()
        except Exception as e:
            logger.error(f"Error closing websocket: {e}")

async def process_and_respond(websocket, stream_sid, audio_data):
    """Process audio chunk and send response without blocking the main loop"""
    try:
        # Transcribe audio
        transcript = await transcribe_audio(audio_data)
        if transcript and transcript != "No transcription available":
            logger.info(f"Processing transcript: {transcript}")
            # Get LLM response
            llm_response = await get_llm_response(transcript)
            if llm_response:
                # Convert response to speech
                audio_payload = await text_to_speech(llm_response)
                if audio_payload:
                    await websocket.send_json({
                        "event": "media",
                        "streamSid": stream_sid,
                        "media": {
                            "payload": audio_payload
                        }
                    })
                    logger.info("Response sent back to client")
    except Exception as e:
        logger.error(f"Error processing audio chunk: {e}", exc_info=True)
# @app.websocket("/media-stream")
# async def handle_media_stream(websocket: WebSocket):
#     """Handle WebSocket connections for the media stream"""
#     logger.info("Client connected")
#     has_seen_media = False
#     message_count = 0
#     await websocket.accept()

#     stream_sid = None
#     audio_buffer = bytearray()
#     audio_processing_tasks = []

#     try:
#         while True:
#             message = await websocket.receive_text()
#             data = json.loads(message)
            
#             if data['event'] == 'connected':
#                 logger.info("Websocket connection established")
#             elif data['event'] == 'start':
#                 stream_sid = data['start']['streamSid']
#                 logger.info(f"Stream started: {stream_sid}")
#             elif data['event'] == 'media':
                
#                 payload = data['media']['payload']
#                 audio_chunk = base64.b64decode(payload)
#                 audio_buffer.extend(audio_chunk)
#                 logger.info("Processing Media Chunks...")

#                 # Process audio in near real-time when buffer reaches a threshold
#                 if len(audio_buffer) >= 16000:  # Adjust threshold as needed
#                     # Create a task to process audio asynchronously
#                     task = asyncio.create_task(process_audio_chunk(
#                         websocket,
#                         stream_sid,
#                         bytes(audio_buffer)
#                     ))
#                     audio_processing_tasks.append(task)

#                     # Clear processed audio from buffer
#                     audio_buffer = bytearray()
#             elif data['event'] == 'stop':
#                 logger.info("Stream stopped")
#                 # Process any remaining audio
#                 if audio_buffer:
#                     await process_audio_chunk(
#                         websocket,
#                         stream_sid,
#                         bytes(audio_buffer)
#                     )
#                 break
#             message_count += 1

#     except WebSocketDisconnect:
#         logger.info("Client disconnected")
#     except Exception as e:
#         logger.error(f"Error in media stream handling: {e}")
#     finally:
#         # Wait for any ongoing audio processing tasks
#         if audio_processing_tasks:
#             await asyncio.gather(*audio_processing_tasks)
#         try:
#             await websocket.close()
#         except Exception as e:
#             logger.error(f"Error closing websocket: {e}")

# async def process_audio_chunk(websocket, stream_sid, audio_data):
#     """Process an audio chunk and send response"""
#     try:
#         # Transcribe audio
#         transcript = await transcribe_audio(audio_data)

#         if transcript:
#             # Get LLM response
#             llm_response = await get_llm_response(transcript)
#             if llm_response:
#                 # Convert response to speech
#                 audio_payload = await text_to_speech(llm_response)
#                 if audio_payload:
#                     await websocket.send_json({
#                         "event": "media",
#                         "streamSid": stream_sid,
#                         "media": {
#                             "payload": audio_payload
#                         }
#                     })
#     except Exception as e:
#         logger.error(f"Error processing audio chunk: {e}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
