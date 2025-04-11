# import os
# import json
# import base64
# import asyncio
# from typing import Any, Dict
# import requests
# import boto3
# from fastapi import FastAPI, WebSocket, Request
# from fastapi.responses import HTMLResponse
# from fastapi.websockets import WebSocketDisconnect
# from twilio.rest import Client
# from twilio.twiml.voice_response import VoiceResponse, Connect
# from dotenv import load_dotenv
# from pydub import AudioSegment
# from io import BytesIO
# from loguru import logger
# import wave
# import os
# from datetime import datetime


# load_dotenv()

# def load_prompt(file_name):
#     dir_path = os.path.dirname(os.path.realpath(__file__))
#     prompt_path = os.path.join(dir_path, 'prompts', f'{file_name}.txt')

#     try:
#         with open(prompt_path, 'r', encoding='utf-8') as file:
#             return file.read().strip()
#     except FileNotFoundError:
#         print(f"Could not find file: {prompt_path}")
#         raise

# # Configuration
# GROQ_API_KEY = os.getenv('GROQ_API_KEY')
# TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
# TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
# TWILIO_PHONE_NUMBER = os.getenv('TWILIO_PHONE_NUMBER')
# NGROK_URL = os.getenv('NGROK_URL')
# PORT = int(os.getenv('PORT', 8000))
# AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
# AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
# AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')

# SYSTEM_MESSAGE = load_prompt('system_prompt')
# AWS_POLLY_VOICE = os.getenv('AWS_POLLY_VOICE', 'Joanna')
# GROQ_MODEL = os.getenv('GROQ_MODEL', 'llama3-70b-8192')

# app = FastAPI()

# if not GROQ_API_KEY:
#     raise ValueError('Missing the Groq API key. Please set it in the .env file.')

# if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN or not TWILIO_PHONE_NUMBER:
#     raise ValueError('Missing Twilio configuration. Please set it in the .env file.')

# if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
#     raise ValueError('Missing AWS configuration. Please set it in the .env file.')

# # Initialize AWS clients
# polly_client = boto3.client(
#     'polly',
#     aws_access_key_id=AWS_ACCESS_KEY_ID,
#     aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
#     region_name=AWS_REGION
# )

# # Store conversation history for each call
# conversations:Dict[Any, Any] = {}

# @app.get("/", response_class=HTMLResponse)
# async def index_page():
#     return {"message": "Twilio Media Stream Server is running!"}

# @app.post("/make-call")
# async def make_call(request: Request):
#     """Make an outgoing call to the specified phone number."""
#     data = await request.json()
#     to_phone_number = data.get("to")
#     if not to_phone_number:
#         return {"error": "Phone number is required"}

#     client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
#     call = client.calls.create(
#         url=f"{NGROK_URL}/outgoing-call",
#         to=to_phone_number,
#         from_=TWILIO_PHONE_NUMBER
#     )
    
#     # Initialize conversation history for this call
#     conversations[call.sid] = []
    
#     return {"call_sid": call.sid}

# @app.api_route("/outgoing-call", methods=["GET", "POST"])
# async def handle_outgoing_call(request: Request):
#     """Handle outgoing call and return TwiML response to connect to Media Stream."""
#     response = VoiceResponse()
#     response.say("Please wait while we connect your call to the AI voice assistant...")
#     response.pause(length=1)
#     response.say("O.K. you can start talking!")
#     connect = Connect()
#     connect.stream(url=f'wss://{request.url.hostname}/media-stream')
#     response.append(connect)
#     return HTMLResponse(content=str(response), media_type="application/xml")

# async def transcribe_audio(audio_content):
#     """Use Groq's Whisper API for transcription"""
#     try:
#         # Save ulaw audio to a temporary file
#         temp_audio_path = "temp_audio.wav"
#         with open(temp_audio_path, "wb") as f:
#             f.write(base64.b64decode(audio_content))
        
#         # Convert uLaw to wav using pydub
#         audio = AudioSegment.from_file(temp_audio_path, format="wav", codec="ulaw")
#         audio.export("temp_pcm.wav", format="wav")
        
#         # Open the converted file
#         with open("temp_pcm.wav", "rb") as audio_file:
#             # Use Groq's Whisper endpoint (replace with actual endpoint)
#             headers = {
#                 "Authorization": f"Bearer {GROQ_API_KEY}"
#             }
#             files = {
#                 "file": ("audio.wav", audio_file, "audio/wav")
#             }
#             response = requests.post(
#                 "https://api.groq.com/v1/audio/transcriptions",
#                 headers=headers,
#                 files=files,
#                 data={"model": "whisper-large-v3"}
#             )
            
#             if response.status_code == 200:
#                 return response.json().get("text", "")
#             else:
#                 print(f"Transcription error: {response.status_code}, {response.text}")
#                 return ""
#     except Exception as e:
#         print(f"Error in transcribe_audio: {e}")
#         return ""
#     finally:
#         # Clean up temp files
#         if os.path.exists("temp_audio.wav"):
#             os.remove("temp_audio.wav")
#         if os.path.exists("temp_pcm.wav"):
#             os.remove("temp_pcm.wav")

# async def get_groq_response(transcription, conversation_history):
#     """Get response from Groq API"""
#     try:
#         headers = {
#             "Authorization": f"Bearer {GROQ_API_KEY}",
#             "Content-Type": "application/json"
#         }
        
#         messages = [{"role": "system", "content": SYSTEM_MESSAGE}]
#         messages.extend(conversation_history)
#         messages.append({"role": "user", "content": transcription})
        
#         data = {
#             "model": GROQ_MODEL,
#             "messages": messages,
#             "temperature": 0.7,
#             "max_tokens": 512
#         }
        
#         response = requests.post(
#             "https://api.groq.com/v1/chat/completions",
#             headers=headers,
#             json=data
#         )
        
#         if response.status_code == 200:
#             response_json = response.json()
#             ai_response = response_json["choices"][0]["message"]["content"]
#             return ai_response
#         else:
#             print(f"Groq API error: {response.status_code}, {response.text}")
#             return "I'm sorry, I couldn't process your request at the moment."
#     except Exception as e:
#         print(f"Error in get_groq_response: {e}")
#         return "I'm sorry, there was an error processing your request."

# async def synthesize_speech(text):
#     """Convert text to speech using AWS Polly"""
#     try:
#         response = polly_client.synthesize_speech(
#             Text=text,
#             OutputFormat='pcm',
#             VoiceId=AWS_POLLY_VOICE,
#             SampleRate='8000'
#         )
        
#         # Get audio stream from Polly response
#         if "AudioStream" in response:
#             # Convert PCM to ulaw for Twilio
#             pcm_audio = response['AudioStream'].read()
#             audio_segment = AudioSegment(
#                 data=pcm_audio,
#                 sample_width=2,  # 16-bit PCM
#                 frame_rate=8000,
#                 channels=1
#             )
            
#             # Convert to ulaw
#             ulaw_io = BytesIO()
#             audio_segment.export(ulaw_io, format="wav", codec="ulaw")
#             ulaw_audio = ulaw_io.getvalue()
            
#             # Encode as base64 for Twilio
#             return base64.b64encode(ulaw_audio).decode('utf-8')
#     except Exception as e:
#         print(f"Error in synthesize_speech: {e}")
#         return None

# @app.websocket("/media-stream")
# async def handle_media_stream(websocket: WebSocket):
#     """Handle WebSocket connections between Twilio and our services."""
#     print("Client connected")
#     await websocket.accept()
    
#     stream_sid = None
#     call_sid = None
#     audio_buffer = []
#     speech_detected = False
#     silence_counter = 0
#     MAX_SILENCE = 20  # Number of silent chunks to wait before processing
#     log_counter = 0  # To avoid excessive logging
    
#     # Create a directory for debug recordings if it doesn't exist
#     os.makedirs("debug_recordings", exist_ok=True)
    
#     # For debug recording
#     debug_audio_bytes = []
#     recording_start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    
#     try:
#         async for message in websocket.iter_text():
#             data = json.loads(message)
            
#             # Log every 50th message to avoid overwhelming the console
#             if log_counter % 50 == 0:
#                 print(f"Received WebSocket message type: {data['event']}")
#             log_counter += 1
            
#             if data['event'] == 'start':
#                 stream_sid = data['start']['streamSid']
#                 call_sid = data['start'].get('callSid')
#                 print(f"Incoming stream has started {stream_sid}, call SID: {call_sid}")
                
#                 # Initialize conversation history if not exists
#                 if call_sid and call_sid not in conversations:
#                     conversations[call_sid] = []
                
#             elif data['event'] == 'media':
#                 # Process incoming audio
#                 audio_payload = data['media']['payload']
#                 audio_bytes = base64.b64decode(audio_payload)
                
#                 # Save raw audio bytes for debug recording
#                 debug_audio_bytes.append(audio_bytes)
                
#                 # Save debug recording every 1000 chunks or about 20 seconds
#                 if len(debug_audio_bytes) % 1000 == 0:
#                     recording_filename = f"debug_recordings/raw_audio_{recording_start_time}_{len(debug_audio_bytes)//1000}.wav"
#                     save_audio_as_wav(debug_audio_bytes, recording_filename)
#                     print(f"Saved debug recording: {recording_filename}")
                
#                 # Basic energy detection
#                 energy = sum(abs(b - 128) for b in audio_bytes) / len(audio_bytes)
                
#                 # Log audio energy level periodically
#                 if log_counter % 100 == 0:
#                     print(f"Audio energy level: {energy}, Speech detected: {speech_detected}, Silence counter: {silence_counter}")
                
#                 if energy > 5:  # Lowered threshold for better detection
#                     if not speech_detected:
#                         print(f"Speech detected! Energy level: {energy}")
#                         # Start a new recording for this speech segment
#                         speech_recording_time = datetime.now().strftime("%Y%m%d_%H%M%S")
#                         speech_audio_bytes = []
                    
#                     speech_detected = True
#                     silence_counter = 0
#                     audio_buffer.append(audio_payload)
                    
#                     # If we're recording speech, add to the speech recording buffer
#                     if speech_detected:
#                         speech_audio_bytes.append(audio_bytes)
                    
#                 elif speech_detected:
#                     silence_counter += 1
#                     audio_buffer.append(audio_payload)
                    
#                     # Add silence to speech recording too
#                     speech_audio_bytes.append(audio_bytes)
                    
#                     # If we detect enough silence after speech, process the audio
#                     if silence_counter >= MAX_SILENCE:
#                         print(f"Processing audio after detecting {silence_counter} silent chunks. Buffer size: {len(audio_buffer)}")
                        
#                         # Save the complete speech segment
#                         speech_filename = f"debug_recordings/speech_{speech_recording_time}.wav"
#                         save_audio_as_wav(speech_audio_bytes, speech_filename)
#                         print(f"Saved speech recording: {speech_filename}")
                        
#                         # Combine audio chunks
#                         combined_audio = ''.join(audio_buffer)
                        
#                         # Process the audio asynchronously
#                         asyncio.create_task(process_audio_and_respond(
#                             combined_audio, 
#                             websocket, 
#                             stream_sid, 
#                             call_sid
#                         ))
                        
#                         # Reset for next utterance
#                         audio_buffer = []
#                         speech_detected = False
#                         silence_counter = 0
            
#             elif data['event'] == 'stop':
#                 print(f"Stream stopped: {data}")
#                 # Save any remaining audio
#                 if debug_audio_bytes:
#                     final_recording = f"debug_recordings/final_recording_{recording_start_time}.wav"
#                     save_audio_as_wav(debug_audio_bytes, final_recording)
#                     print(f"Saved final recording: {final_recording}")
                        
#     except WebSocketDisconnect:
#         print("Client disconnected.")
#         # Save any remaining audio on disconnect
#         if debug_audio_bytes:
#             final_recording = f"debug_recordings/disconnect_recording_{recording_start_time}.wav"
#             save_audio_as_wav(debug_audio_bytes, final_recording)
#             print(f"Saved disconnect recording: {final_recording}")
#     except Exception as e:
#         print(f"Error in handle_media_stream: {e}")
#         import traceback
#         traceback.print_exc()

# def save_audio_as_wav(audio_bytes_list, filename):
#     """Save raw audio bytes as a WAV file"""
#     try:
#         # Twilio sends ulaw 8kHz audio, but we'll save as PCM for compatibility
#         with wave.open(filename, 'wb') as wav_file:
#             wav_file.setnchannels(1)  # Mono
#             wav_file.setsampwidth(2)  # 16-bit PCM
#             wav_file.setframerate(8000)  # 8kHz
            
#             # Convert ulaw to PCM or save as raw without compression specification
#             wav_data = b''.join(audio_bytes_list)
            
#             # Option 1: Save as raw PCM without trying to specify compression
#             wav_file.writeframes(wav_data)
            
#         return True
#     except Exception as e:
#         print(f"Error saving WAV file: {e}")
#         import traceback
#         traceback.print_exc()
        
#         # Fallback: save as raw binary file if WAV conversion fails
#         try:
#             raw_filename = filename.replace('.wav', '.raw')
#             with open(raw_filename, 'wb') as raw_file:
#                 raw_file.write(b''.join(audio_bytes_list))
#             print(f"Saved as raw binary file instead: {raw_filename}")
#             return True
#         except Exception as e2:
#             print(f"Error saving raw file: {e2}")
#             return False

# async def process_audio_and_respond(audio_content, websocket, stream_sid, call_sid):
#     """Process audio and send response back to the caller"""
#     try:
#         # Step 1: Transcribe the audio
#         transcription = await transcribe_audio(audio_content)
#         if not transcription.strip():
#             return
            
#         print(f"Transcription: {transcription}")
        
#         # Step 2: Get LLM response from Groq
#         conversation_history = conversations.get(call_sid, [])
#         ai_response = await get_groq_response(transcription, conversation_history)
#         print(f"AI Response: {ai_response}")
        
#         # Update conversation history
#         if call_sid:
#             conversations[call_sid].append({"role": "user", "content": transcription})
#             conversations[call_sid].append({"role": "assistant", "content": ai_response})
#             # Keep history manageable (last 10 exchanges)
#             if len(conversations[call_sid]) > 20:
#                 conversations[call_sid] = conversations[call_sid][-20:]
        
#         # Step 3: Synthesize speech
#         audio_payload = await synthesize_speech(ai_response)
#         if not audio_payload:
#             return
            
#         # Step 4: Send audio back to Twilio
#         audio_delta = {
#             "event": "media",
#             "streamSid": stream_sid,
#             "media": {
#                 "payload": audio_payload
#             }
#         }
#         await websocket.send_json(audio_delta)
        
#     except Exception as e:
#         print(f"Error in process_audio_and_respond: {e}")

# if __name__ == "__main__":
#     import uvicorn
#     to_phone_number = "+917011897710"
#     client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
#     try:
#         call = client.calls.create(
#             url=f"{NGROK_URL}/outgoing-call",
#             to=to_phone_number,
#             from_=TWILIO_PHONE_NUMBER
#         )
#         print(f"Call initiated with SID: {call.sid}")
#         # Initialize conversation for this call
#         conversations[call.sid] = []
#     except Exception as e:
#         print(f"Error initiating call: {e}")
#     uvicorn.run(app, host="0.0.0.0", port=PORT)

@app.get("/test-voice")
async def test_voice():
    """Test endpoint to verify the voice pipeline works."""
    try:
        # Create a test processor
        processor = AudioStreamProcessor()
        session_id = await processor.create_session()
        
        # Test transcription
        transcript = processor._transcribe_audio(generate_test_audio())
        logger.info(f"Test transcription: {transcript}")
        
        # Test LLM
        response = processor._generate_llm_response(
            system_message="You are a helpful assistant.",
            conversation=[{"role": "user", "content": "Hello, how are you?"}],
            temperature=0.7
        )
        logger.info(f"Test LLM response: {response}")
        
        # Test TTS
        test_audio = await processor._generate_speech(session_id=session_id, text="This is a test of the speech synthesis system.")
        
        return {
            "status": "success",
            "transcription": transcript,
            "llm_response": response,
            "audio_generated": len(test_audio) > 0
        }
    except Exception as e:
        logger.error(f"Test error: {e}")
        return {"status": "error", "message": str(e)}

def generate_test_audio():
    """Generate a test audio sample."""
    from pydub.generators import Sine
    
    # Generate a test tone
    sine_wave = Sine(440)  # 440 Hz
    audio = sine_wave.to_audio_segment(duration=1000)  # 1 second
    audio = audio.set_channels(1).set_frame_rate(8000).set_sample_width(2)
    
    buffer = BytesIO()
    audio.export(buffer, format="wav")
    buffer.seek(0)
    return buffer.read()