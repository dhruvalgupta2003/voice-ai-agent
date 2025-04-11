import asyncio
import base64
import glob
import os
import threading
import time
from io import BytesIO
from queue import Empty, Queue
from typing import Dict
from uuid import uuid4
from datetime import datetime, timezone

import boto3
from fastapi import WebSocketDisconnect
import numpy as np
import pywav
import requests
from pydub import AudioSegment

from config.load_config import Config
from utils.load_prompt import load_prompt
from utils.logger import get_logger
from utils.validate_config import validate_config

# Load Config
config = Config()

# setup logging
logger = get_logger()

# Config Validation
validate_config(config)

# Load system prompt
SYSTEM_MESSAGE = load_prompt('loan_repayment_prompt')

# Initialize AWS Polly client
polly_client = boto3.client(
    "polly",
    aws_access_key_id=config.AWS_ACCESS_KEY,
    aws_secret_access_key=config.AWS_SECRET_KEY,
    region_name=config.AWS_REGION,
)

# Audio duration for chunks
MEDIA_INTERVAL_SECONDS = 5

session_meta:Dict = {}  # session_id -> {'buffer': [], 'start_time': datetime}


class AudioStreamProcessor:
    """Process audio streams between clients and AI services."""

    def __init__(self):
        self.active_sessions = {}
        self.lock = threading.RLock() # Add thread-safe lock for session operations

    async def create_session(self, client_ws):
        """Create a new audio processing session."""
        try:
            session_id = str(uuid4())

            session = {
                "id": session_id,
                "client_ws": client_ws,
                "stream_sid": '',
                "audio_buffer": bytearray(),
                "transcription_queue": Queue(),
                "response_queue": Queue(),
                "audio_chunk_queue": Queue(),
                "is_speaking": False,
                "last_speech_time": time.time(),
                "processing_status": "active",  # Add this line to initialize the status
                "config": {
                    "voice": config.VOICE,
                    "instructions": SYSTEM_MESSAGE,
                    "temperature": 0.7,
                },
                "loop": asyncio.get_event_loop(),
            }
            
            with self.lock:  # Use the lock when modifying shared data
                self.active_sessions[session_id] = session

            # Start processing threads for this session
            threading.Thread(
                target=self._transcription_worker, args=(session_id,), daemon=True
            ).start()
            threading.Thread(
                target=self._llm_worker, args=(session_id,), daemon=True
            ).start()
            threading.Thread(
                target=self._tts_worker, args=(session_id,), daemon=True
            ).start()

            # Notify client about session creation
            await client_ws.send_json(
                {"type": "session.created", "session": {"id": session_id}}
            )
            logger.info(f"Session created: {session_id}")
            return session_id
        except Exception as e:
            logger.error(f"Failed to create session: {str(e)}")
            # Send error to client
            try:
                await client_ws.send_json({"type": "error", "message": "Failed to create session"})
            except Exception as e:
                logger.error(f"Exception sending error to client: {str(e)}")
            raise

    def close_session(self, session_id):
        """Close and clean up a session."""
        try:
            with self.lock:
                if session_id in self.active_sessions:
                    logger.info(f"Closing session: {session_id}")
                    # Set status to closing to prevent new processing
                    self.active_sessions[session_id]["processing_status"] = "closing"
                    
                    # Signal threads to stop
                    self.active_sessions[session_id]["transcription_queue"].put(None)
                    self.active_sessions[session_id]["response_queue"].put(None)
                    self.active_sessions[session_id]["audio_chunk_queue"].put(None)
                    
                    # Clean up temp files
                    temp_pattern = f"temp_audio/{session_id}_*.wav"
                    for file in glob.glob(temp_pattern):
                        try:
                            os.remove(file)
                            logger.debug(f"Removed temp file: {file}")
                        except Exception as e:
                            logger.warning(f"Failed to remove temp file {file}: {e}")
                    
                    # Remove from active sessions
                    del self.active_sessions[session_id]
                    logger.info(f"Session {session_id} closed and cleaned up")
                else:
                    logger.warning(f"Attempted to close non-existent session: {session_id}")
        except Exception as e:
            logger.error(f"Error closing session {session_id}: {e}")


    async def handle_media_batch(self, session_id, payload_batch, stream_sid):
        """Process accumulated audio chunks"""
        try:
            with self.lock:
                if session_id not in self.active_sessions:
                    logger.warning(f"Attempted to process media for non-existent session: {session_id}")
                    return
                
                # Check if processing_status exists (for backward compatibility)
                if "processing_status" in self.active_sessions[session_id] and \
                   self.active_sessions[session_id]["processing_status"] != "active":
                    logger.info(f"Skipping media processing for closing session: {session_id}")
                    return

            # Store reference to session outside lock to reduce contention
            session = self.active_sessions[session_id]
            
            # Update session metadata
            meta = session_meta.setdefault(session_id, {
                'buffer': [],
                'start_time': datetime.now(timezone.utc),
                'last_activity': datetime.now(timezone.utc)
            })
            meta['last_activity'] = datetime.now(timezone.utc)
            
            # Update stream SID if needed
            if stream_sid and not session['stream_sid']:
                session['stream_sid'] = stream_sid
                logger.info(f"Updated stream SID for session {session_id}: {stream_sid}")

            # Process full batch
            try:
                audio_bytes = b''.join(payload_batch)
                audio_size = len(audio_bytes)
                logger.debug(f"Processing {audio_size} bytes of audio for session {session_id}")
                
                # Check for voice activity (simplified)
                is_silent = self._is_silent(audio_bytes)
                if is_silent:
                    session["is_speaking"] = False
                else:
                    session["is_speaking"] = True
                    session["last_speech_time"] = time.time()
                
                # Queue audio for transcription
                session["transcription_queue"].put(audio_bytes)
            except Exception as e:
                logger.error(f"Error processing audio batch for session {session_id}: {e}")
        except Exception as e:
            logger.error(f"Error in handle_media_batch for session {session_id}: {e}")
    
    def _is_silent(self, audio_bytes):
        """Simple detection of silence in audio bytes.
        Returns True if audio appears to be silent."""
        try:
            # Simple amplitude check - could be improved with proper VAD
            if not audio_bytes:
                return True
            
            # Convert to numpy array for processing
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
            
            # Calculate RMS amplitude
            rms = np.sqrt(np.mean(np.square(audio_array)))
            
            # Threshold for silence
            return rms < 500  # Adjust this threshold based on your audio characteristics
        except Exception as e:
            logger.warning(f"Error detecting silence: {e}")
            return False  # Default to non-silence on error
    
    
    async def get_websocket(self, session_id):
        """Get the websocket for a session."""
        try:
            with self.lock:
                if session_id in self.active_sessions:
                    return self.active_sessions[session_id]["client_ws"]
            return None
        except Exception as e:
            logger.error(f"Error getting websocket for session {session_id}: {e}")
            return None

    def _transcription_worker(self, session_id):
        """Worker thread for transcribing audio using Groq's Whisper."""
        try:
            # Check if session still exists
            if session_id not in self.active_sessions:
                logger.warning(f"Transcription worker started for non-existent session: {session_id}")
                return
                
            session = self.active_sessions[session_id]
            logger.info(f"Transcription worker started for session: {session_id}")

            # Buffer for collecting audio chunks for transcription
            transcription_buffer = bytearray()
            # trunk-ignore(ruff/F841)
            silence_duration = 0
            last_transcription_time = time.time()

            while True:
                try:
                    # Get next audio chunk with timeout
                    try:
                        chunk = session["transcription_queue"].get(timeout=1.0)
                    except Empty:
                        # Check if we should process buffer due to silence
                        current_time = time.time()
                        if (current_time - session["last_speech_time"] > 1.0 and 
                            transcription_buffer and 
                            current_time - last_transcription_time > 2.0):
                            # Process what we have if silence detected
                            logger.debug(f"Processing {len(transcription_buffer)} bytes after silence detected")
                            transcript = self._transcribe_audio(transcription_buffer)
                            if transcript:
                                session["response_queue"].put(transcript)
                                last_transcription_time = current_time
                            transcription_buffer = bytearray()
                        continue
                        
                    if chunk is None:  # Termination signal
                        logger.info(f"Transcription worker for session {session_id} received termination signal")
                        break

                    transcription_buffer.extend(chunk)

                    # Process if buffer is large enough or speech has stopped
                    buffer_size = len(transcription_buffer)
                    current_time = time.time()
                    time_since_last_speech = current_time - session["last_speech_time"]
                    
                    # Process buffer if:
                    # 1. We have enough audio data OR
                    # 2. Speech has stopped for more than 0.8 seconds
                    if buffer_size >= 32000 or (buffer_size > 8000 and time_since_last_speech > 0.8):
                        if transcription_buffer:
                            # Transcribe using Groq's Whisper API
                            logger.debug(f"Transcribing {buffer_size} bytes of audio for session {session_id}")
                            transcript = self._transcribe_audio(transcription_buffer)
                            if transcript:
                                # Put transcript into LLM queue
                                session["response_queue"].put(transcript)
                                last_transcription_time = current_time
                                logger.info(f"Session {session_id} transcribed: '{transcript}'")

                            # Clear buffer after processing
                            transcription_buffer = bytearray()

                    session["transcription_queue"].task_done()
                except Exception as e:
                    logger.error(f"Error in transcription worker for session {session_id}: {e}")
                    # Continue processing
            
            # Process any remaining audio in buffer before exiting
            if transcription_buffer:
                try:
                    transcript = self._transcribe_audio(transcription_buffer)
                    if transcript:
                        session["response_queue"].put(transcript)
                except Exception as e:
                    logger.error(f"Final transcription error: {e}")
                    
            logger.info(f"Transcription worker for session {session_id} exiting")
                    
        except Exception as e:
            logger.error(f"Fatal error in transcription worker for session {session_id}: {e}")

    def _llm_worker(self, session_id):
        """Worker thread for processing text with Groq LLM."""
        try:
            if session_id not in self.active_sessions:
                logger.warning(f"LLM worker started for non-existent session: {session_id}")
                return
                
            session = self.active_sessions[session_id]
            logger.info(f"LLM worker started for session: {session_id}")
            conversation_history = []

            while True:
                try:
                    # Get transcript with timeout
                    try:
                        transcript = session["response_queue"].get(timeout=1.0)
                    except Empty:
                        continue
                        
                    if transcript is None:  # Termination signal
                        logger.info(f"LLM worker for session {session_id} received termination signal")
                        break

                    # Skip empty or whitespace-only transcripts
                    if not transcript or transcript.strip() == "":
                        session["response_queue"].task_done()
                        continue

                    logger.info(f"Processing transcript for session {session_id}: '{transcript}'")

                    # Add user message to history
                    conversation_history.append({"role": "user", "content": transcript})
                    
                    # Keep conversation history manageable (last 10 messages)
                    if len(conversation_history) > 10:
                        conversation_history = conversation_history[-10:]

                    # Process with Groq LLM
                    response = self._generate_llm_response(
                        system_message=session["config"]["instructions"],
                        conversation=conversation_history,
                        temperature=session["config"]["temperature"],
                    )

                    # Add assistant message to history
                    conversation_history.append({"role": "assistant", "content": response})

                    # Send text response event first
                    asyncio.run_coroutine_threadsafe(
                        session["client_ws"].send_json(
                            {"type": "response.content.delta", "delta": response}
                        ),
                        session["loop"]
                    )

                    # Generate speech from response
                    audio_path = self._generate_speech_sync(session_id, response)

                    if audio_path:
                        # Send audio response
                        asyncio.run_coroutine_threadsafe(
                            self._send_audio_response(session["client_ws"], audio_path, response, session_id),
                            session["loop"]
                        )

                    # Signal completion
                    asyncio.run_coroutine_threadsafe(
                        session["client_ws"].send_json({"type": "response.content.done"}),
                        session["loop"]
                    )
                    
                    logger.info(f"LLM processing complete for session {session_id}")
                    
                except Exception as e:
                    logger.error(f"Error in LLM worker for session {session_id}: {e}")
                    # Try to send error message to client
                    try:
                        error_msg = "I'm sorry, I encountered an error processing your message."
                        asyncio.run_coroutine_threadsafe(
                            session["client_ws"].send_json({"type": "error", "message": error_msg}),
                            session["loop"]
                        )
                    except Exception:
                        pass
                
                finally:
                    try:
                        session["response_queue"].task_done()
                    except Exception:
                        pass
            
            logger.info(f"LLM worker for session {session_id} exiting")
                    
        except Exception as e:
            logger.error(f"Fatal error in LLM worker for session {session_id}: {e}")
    
    async def _send_audio_response(self, websocket, audio_path, text, session_id):
        """Send audio response to client."""
        try:
            if not audio_path or not os.path.exists(audio_path):
                logger.error(f"Audio file not found: {audio_path}")
                return
                
            with open(audio_path, 'rb') as f:
                audio_bytes = f.read()
                
            logger.debug(f"Sending {len(audio_bytes)} bytes of audio for session {session_id}")
            encoded_audio = base64.b64encode(audio_bytes).decode('utf-8')

            await websocket.send_json({
                "event": "media",
                "streamSid": self.active_sessions[session_id]['stream_sid'],
                "text": text,
                "audio": encoded_audio
            })
            logger.info(f"Audio response sent to client for session {session_id}")
            
            # Delete the temporary file after sending
            try:
                os.remove(audio_path)
                logger.debug(f"Removed temporary audio file: {audio_path}")
            except Exception as e:
                logger.warning(f"Failed to remove temporary file {audio_path}: {e}")
                
        except WebSocketDisconnect:
            logger.warning(f"Client disconnected when sending audio for session {session_id}")
        except Exception as e:
            logger.error(f"Error sending audio response for session {session_id}: {e}")

    def _tts_worker(self, session_id):
        """Worker thread for sending audio chunks back to client."""
        try:
            if session_id not in self.active_sessions:
                logger.warning(f"TTS worker started for non-existent session: {session_id}")
                return
                
            session = self.active_sessions[session_id]
            logger.info(f"TTS worker started for session: {session_id}")

            while True:
                try:
                    # Get audio chunk with timeout
                    try:
                        audio_chunk = session["audio_chunk_queue"].get(timeout=1.0)
                    except Empty:
                        continue
                        
                    if audio_chunk is None:  # Termination signal
                        logger.info(f"TTS worker for session {session_id} received termination signal")
                        break

                    # Send audio chunk to client
                    encoded_chunk = base64.b64encode(audio_chunk).decode("utf-8")

                    asyncio.run_coroutine_threadsafe(
                        session["client_ws"].send_json(
                            {"type": "response.audio.delta", "delta": encoded_chunk}
                        ),
                        session["loop"]
                    )
                    
                    logger.debug(f"Audio chunk sent for session {session_id}")
                except Exception as e:
                    logger.error(f"Error in TTS worker for session {session_id}: {e}")
                finally:
                    try:
                        session["audio_chunk_queue"].task_done()
                    except Exception:
                        pass
            
            logger.info(f"TTS worker for session {session_id} exiting")
            
        except Exception as e:
            logger.error(f"Fatal error in TTS worker for session {session_id}: {e}")

    def _generate_speech_sync(self, session_id, text):
        """Generate speech using AWS Polly and save to temp file."""
        start_time = time.time()
        try:
            if not text or text.strip() == "":
                logger.warning(f"Empty text provided for speech generation for session {session_id}")
                return None
                
            # Create temp directory if it doesn't exist
            os.makedirs("temp_audio", exist_ok=True)
            
            # Generate a unique filename for this audio
            temp_path = f"temp_audio/{session_id}_{int(time.time())}.wav"
            
            # Split text into sentences for more natural speech
            sentences = self._split_into_sentences(text)
            if not sentences:
                logger.warning(f"No sentences extracted for TTS in session {session_id}")
                return None
                
            logger.info(f"Generating speech for {len(sentences)} sentences in session {session_id}")
            
            # Create an empty audio segment
            combined_audio = AudioSegment.empty()
            
            for i, sentence in enumerate(sentences):
                if not sentence.strip():
                    continue
                
                try:
                    # Generate speech with AWS Polly
                    response = polly_client.synthesize_speech(
                        Text=sentence,
                        OutputFormat="mp3",  # Use mp3 for better quality
                        VoiceId=self.active_sessions[session_id]["config"]["voice"],
                    )
                    
                    # Get audio data from the response
                    audio_stream = response["AudioStream"].read()
                    
                    # Convert to AudioSegment
                    sentence_audio = AudioSegment.from_mp3(BytesIO(audio_stream))
                    
                    # Add to combined audio
                    combined_audio += sentence_audio
                    
                    # Add a small pause between sentences for natural rhythm
                    if i < len(sentences) - 1:
                        combined_audio += AudioSegment.silent(duration=150)
                        
                except Exception as e:
                    logger.error(f"Error generating speech for sentence in session {session_id}: {e}")
                    # Continue with next sentence
            
            # Export as WAV file
            combined_audio.export(temp_path, format="wav")
            
            duration = time.time() - start_time
            logger.info(f"Generated speech saved to {temp_path} in {duration:.2f}s")
            
            return temp_path
            
        except Exception as e:
            logger.error(f"Speech generation error for session {session_id}: {e}")
            return None

    async def _transcribe_audio(self, audio_data):
        """Transcribe audio using Groq's Whisper API."""
        if not audio_data:
            return ""

        start_time = time.time()
        try:
            raw_data = b"".join(audio_data)

            # Write µ-law WAV audio to a file using pywav
            filepath = "audio.wav"
            wave_writer = pywav.WavWrite(filepath, 1, 8000, 8, 7)  # mono, 8000Hz, 8bit, µ-law
            wave_writer.write(raw_data)
            wave_writer.close()

            # Open the written file for reading as binary
            with open(filepath, "rb") as audio_file:
                headers = {"Authorization": f"Bearer {config.GROQ_API_KEY}"}
                files = {"file": ("audio.wav", audio_file, "audio/wav")}

                response = requests.post(
                    "https://api.groq.com/openai/v1/audio/transcriptions",
                    headers=headers,
                    files=files,
                    data={"model": "whisper-large-v3"},
                    timeout=10
                )

            if response.status_code == 200:
                transcribed_text = response.json().get("text", "")
                duration = time.time() - start_time
                logger.info(f"TRANSCRIBED TEXT in {duration:.2f}s: {transcribed_text}")
                return transcribed_text
            else:
                logger.error(f"Transcription API error: {response.status_code} {response.text}")
                return ""

        except requests.exceptions.Timeout:
            logger.error("Transcription request timed out")
            return ""
        except requests.exceptions.RequestException as e:
            logger.error(f"Transcription request error: {e}")
            return ""
        except Exception as e:
            logger.error(f"Transcription processing error: {e}")
            return ""


    async def _generate_llm_response(self, system_message, conversation, temperature=0.7):
        """Generate response using Groq LLM API."""
        start_time = time.time()
        try:
            headers = {
                "Authorization": f"Bearer {config.GROQ_API_KEY}",
                "Content-Type": "application/json",
            }

            messages = [{"role": "system", "content": system_message}]
            messages.extend(conversation)

            payload = {
                "model": "llama3-8b-8192",  # Groq's Llama 3 model
                "messages": messages,
                "temperature": temperature,
                "max_tokens": 800,  # Add token limit
            }

            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=15
            )

            if response.status_code == 200:
                llm_response = response.json()["choices"][0]["message"]["content"]
                duration = time.time() - start_time
                logger.info(f"Response Generated in {duration:.2f}s")
                return llm_response
            else:
                logger.error(f"LLM API error: {response.status_code} {response.text}")
                return "I'm sorry, I couldn't process your request."
        except requests.exceptions.Timeout:
            logger.error("LLM request timed out")
            return "I'm sorry, the response took too long to generate. Please try again."
        except requests.exceptions.RequestException as e:
            logger.error(f"LLM request error: {e}")
            return "I'm sorry, there was a problem with the AI service. Please try again."
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            return "I'm sorry, I'm having trouble generating a response."

    def _split_into_sentences(self, text):
        """Split text into sentences for better TTS chunking."""
        try:
            import re

            # More sophisticated sentence splitting with handling for abbreviations
            # This could be improved further with NLP libraries
            text = text.replace('\n', ' ')  # Replace newlines with spaces
            
            # Pattern handles common sentence endings while preserving abbreviations like "Dr."
            pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s'
            sentences = re.split(pattern, text)
            
            # Clean up sentences
            sentences = [s.strip() for s in sentences if s.strip()]
            
            return sentences
        except Exception as e:
            logger.error(f"Error splitting text into sentences: {e}")
            # Fallback to simpler approach
            return [text]