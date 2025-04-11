# # import asyncio
# # import websockets
# # import json
# # import base64
# # import wave
# # import pyaudio

# # async def send_audio_stream(uri):
# #     """
# #     Simulate sending audio stream to the WebSocket server
# #     This example uses a sample WAV file or generates a sine wave
# #     """
# #     try:
# #         async with websockets.connect(uri) as websocket:
# #             # Simulate stream start event
# #             await websocket.send(json.dumps({
# #                 "event": "start",
# #                 "start": {
# #                     "streamSid": "test-stream-123"
# #                 }
# #             }))

# #             # Option 1: Load a WAV file
# #             try:
# #                 with wave.open('sample.wav', 'rb') as wf:
# #                     # Read audio data
# #                     frames = wf.readframes(wf.getnframes())
# #                     audio_chunks = [frames[i:i+1024] for i in range(0, len(frames), 1024)]
                    
# #                     for chunk in audio_chunks:
# #                         # Simulate media event
# #                         await websocket.send(json.dumps({
# #                             "event": "media",
# #                             "media": {
# #                                 "payload": base64.b64encode(chunk).decode('utf-8')
# #                             }
# #                         }))
# #                         await asyncio.sleep(0.1)  # Simulate streaming delay
            
# #             except FileNotFoundError:
# #                 # Option 2: Generate a simple sine wave if no WAV file
# #                 p = pyaudio.PyAudio()
# #                 stream = p.open(format=pyaudio.paFloat32,
# #                                 channels=1,
# #                                 rate=44100,
# #                                 output=True)
                
# #                 # Generate simple sine wave
# #                 import numpy as np
# #                 sample_rate = 44100
# #                 duration = 5  # 5 seconds
# #                 frequency = 440  # A4 note
                
# #                 t = np.linspace(0, duration, int(sample_rate * duration), False)
# #                 tone = np.sin(2 * np.pi * frequency * t)
                
# #                 # Convert to bytes and send in chunks
# #                 for i in range(0, len(tone), 1024):
# #                     chunk = tone[i:i+1024]
# #                     chunk_bytes = chunk.astype(np.float32).tobytes()
                    
# #                     await websocket.send(json.dumps({
# #                         "event": "media",
# #                         "media": {
# #                             "payload": base64.b64encode(chunk_bytes).decode('utf-8')
# #                         }
# #                     }))
# #                     await asyncio.sleep(0.1)

# #             # Simulate stream stop event
# #             await websocket.send(json.dumps({
# #                 "event": "stop"
# #             }))

# #             # Wait for server response
# #             response = await websocket.recv()
# #             print("Server response:", response)

# #     except Exception as e:
# #         print(f"Error: {e}")

# # async def main():
# #     # Replace with your actual WebSocket server URL
# #     uri = "ws://localhost:8080/media-stream"
# #     await send_audio_stream(uri)

# # if __name__ == "__main__":
# #     asyncio.run(main())

# import asyncio
# import websockets
# import json
# import base64
# import numpy as np
# import io
# import wave

# async def generate_audio_data():
#     """
#     Generate audio data as a WAV file-like byte stream
#     """
#     # Generate a simple sine wave
#     sample_rate = 16000  # Use a standard speech recognition sample rate
#     duration = 5  # 5 seconds
#     frequency = 440  # A4 note

#     t = np.linspace(0, duration, int(sample_rate * duration), False)
#     tone = np.sin(2 * np.pi * frequency * t)
    
#     # Convert to 16-bit PCM
#     audio = (tone * 32767).astype(np.int16)
    
#     # Create a byte stream simulating a WAV file
#     wav_buffer = io.BytesIO()
#     with wave.open(wav_buffer, 'wb') as wav_file:
#         wav_file.setnchannels(1)  # mono
#         wav_file.setsampwidth(2)  # 16-bit
#         wav_file.setframerate(sample_rate)
#         wav_file.writeframes(audio.tobytes())
    
#     wav_buffer.seek(0)
#     return wav_buffer.read()

# async def send_audio_stream(uri):
#     """
#     Simulate sending audio stream to the WebSocket server
#     """
#     try:
#         async with websockets.connect(uri) as websocket:
#             # Simulate stream start event
#             await websocket.send(json.dumps({
#                 "event": "start",
#                 "start": {
#                     "streamSid": "test-stream-123"
#                 }
#             }))

#             # Generate audio data
#             audio_data = await generate_audio_data()
            
#             # Send audio in chunks
#             chunk_size = 1024
#             for i in range(0, len(audio_data), chunk_size):
#                 chunk = audio_data[i:i+chunk_size]
                
#                 await websocket.send(json.dumps({
#                     "event": "media",
#                     "media": {
#                         "payload": base64.b64encode(chunk).decode('utf-8')
#                     }
#                 }))
#                 await asyncio.sleep(0.1)

#             # Simulate stream stop event
#             await websocket.send(json.dumps({
#                 "event": "stop"
#             }))

#             # Wait for server response
#             try:
#                 response = await websocket.recv()
#                 print("Server response:", response)
#             except Exception as e:
#                 print("No response received:", e)

#     except Exception as e:
#         print(f"Error: {e}")

# async def main():
#     # Replace with your actual WebSocket server URL
#     uri = "ws://localhost:8080/media-stream"
#     await send_audio_stream(uri)

# if __name__ == "__main__":
#     asyncio.run(main())


# import asyncio
# import websockets
# import json
# import base64

# async def send_audio_stream(uri, file_path):
#     """Send audio stream from WAV file to WebSocket server"""
#     try:
#         async with websockets.connect(uri) as websocket:
#             # Read entire file first (for simplicity)
#             with open(file_path, 'rb') as f:
#                 audio_data = f.read()

#             # Send stream start event
#             await websocket.send(json.dumps({
#                 "event": "start",
#                 "start": {
#                     "streamSid": "test-stream-123"
#                 }
#             }))

#             # Send audio in chunks
#             chunk_size = 1024
#             for i in range(0, len(audio_data), chunk_size):
#                 chunk = audio_data[i:i+chunk_size]
#                 await websocket.send(json.dumps({
#                     "event": "media",
#                     "media": {
#                         "payload": base64.b64encode(chunk).decode('utf-8')
#                     }
#                 }))
#                 await asyncio.sleep(0.1)

#             # Send stream stop event
#             await websocket.send(json.dumps({
#                 "event": "stop"
#             }))

#             # Wait for server response
#             try:
#                 response = await websocket.recv()
#                 print("Server response:", response)
#             except Exception as e:
#                 print("No response received:", e)

#     except Exception as e:
#         print(f"Error: {e}")

# async def main():
#     uri = "ws://localhost:8080/media-stream"
#     audio_file = "sample.wav"  # Your WAV file path
#     await send_audio_stream(uri, audio_file)

# if __name__ == "__main__":
#     asyncio.run(main())

import asyncio
import websockets
import json
import base64
import sounddevice as sd
# import numpy as np
from scipy.io.wavfile import write

async def record_audio(file_path, duration=5):
    """Record audio from the microphone and save it as a WAV file."""
    print("Recording...")
    fs = 44100  # Sample rate
    audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()  # Wait until recording is finished
    write(file_path, fs, audio_data)  # Save as WAV file
    print("Recording finished.")

async def send_audio_stream(uri, file_path):
    """Send audio stream from WAV file to WebSocket server."""
    try:
        async with websockets.connect(uri) as websocket:
            # Read entire file first (for simplicity)
            with open(file_path, 'rb') as f:
                audio_data = f.read()

            # Send stream start event
            await websocket.send(json.dumps({
                "event": "start",
                "start": {
                    "streamSid": "test-stream-123"
                }
            }))

            # Send audio in chunks
            chunk_size = 1024
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i:i+chunk_size]
                await websocket.send(json.dumps({
                    "event": "media",
                    "media": {
                        "payload": base64.b64encode(chunk).decode('utf-8')
                    }
                }))
                await asyncio.sleep(0.1)

            # Send stream stop event
            await websocket.send(json.dumps({
                "event": "stop"
            }))

            # Wait for server response
            try:
                response = await websocket.recv()
                print("Server response:", response)
            except Exception as e:
                print("No response received:", e)

    except Exception as e:
        print(f"Error: {e}")

async def main():
    # uri = "ws://localhost:8000/media-stream" # replit main.py
    uri = "ws://localhost:8080/media-stream" # main.py
    audio_file = "sample.wav"  # Your WAV file path

    # Record audio before sending it
    await record_audio(audio_file, duration=5)  # Record for 5 seconds

    # Send the recorded audio over WebSocket
    await send_audio_stream(uri, audio_file)

if __name__ == "__main__":
    asyncio.run(main())
