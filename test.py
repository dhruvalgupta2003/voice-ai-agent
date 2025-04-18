# # from datetime import datetime
# # from utils.load_prompt import load_prompt
# # from config.load_config import Config

# # config = Config()

# # SYSTEM_MESSAGE= load_prompt('loan_repayment_prompt')

# # data:dict = {
# #     "customer_name": "Michael",
# #     "account_number": "12345678",
# #     "loan_amount": "10000",
# #     "due_date": "15th April 2025"
# # }

# # # Get loan account details - in production this would come from your CRM/database
# # customer_name = data.get("customer_name", "Michael")
# # account_number = data.get("account_number", "1234")
# # loan_amount = data.get("loan_amount", "$10,000")
# # due_date = data.get("due_date", "15th April 2025")
# # parsed_due_date = datetime.strptime(data.get("due_date", "15th April 2025"), "%dth %B %Y").date()

# # # Determine overdue or upcoming
# # today = datetime.today().date()
# # overdue_or_upcoming = "Overdue" if parsed_due_date < today else "Upcoming"


# # # Create customized system prompt with call details
# # custom_prompt = SYSTEM_MESSAGE.replace("[Customer Name]", customer_name)
# # custom_prompt = custom_prompt.replace("XXXX", account_number[-4:] if len(account_number) >= 4 else account_number)
# # custom_prompt = custom_prompt.replace("₹XX,XXX", f"${loan_amount}")
# # custom_prompt = custom_prompt.replace("[Date]", due_date)
# # custom_prompt = custom_prompt.replace("[Agent Name]", config.AGENT_NAME)
# # custom_prompt = custom_prompt.replace("[Company Name]", config.COMPANY_NAME)
# # custom_prompt = custom_prompt.replace("[Overdue/Upcoming in X days]", overdue_or_upcoming)

# # print(custom_prompt)

# # from typing import Dict, List


# # response = {
# #     "type": "response.done",
# #     "event_id": "event_BKkyxH7qKdFzWW5gU9o16",
# #     "response": {
# #         "object": "realtime.response",
# #         "id": "resp_BKkyvYrRVCILg4qVAO4BB",
# #         "status": "completed",
# #         "status_details": None,
# #         "output": [
# #             {
# #                 "id": "item_BKkyvi2BwdT0aMwU141Eg",
# #                 "object": "realtime.item",
# #                 "type": "message",
# #                 "status": "completed",
# #                 "role": "assistant",
# #                 "content": [
# #                     {
# #                         "type": "audio",
# #                         "transcript": "Hello, this is Ava calling on behalf of OakStone-Mortgages regarding your loan account ending with 1234. Is this Michael?",
# #                     }
# #                 ],
# #             }
# #         ],
# #         "conversation_id": "conv_BKkyuRrDYuCmbDJ4d7tQT",
# #         "modalities": ["audio", "text"],
# #         "voice": "alloy",
# #         "output_audio_format": "g711_ulaw",
# #         "temperature": 0.8,
# #         "max_output_tokens": "inf",
# #         "usage": {
# #             "total_tokens": 749,
# #             "input_tokens": 536,
# #             "output_tokens": 213,
# #             "input_token_details": {
# #                 "text_tokens": 528,
# #                 "audio_tokens": 8,
# #                 "cached_tokens": 0,
# #                 "cached_tokens_details": {"text_tokens": 0, "audio_tokens": 0},
# #             },
# #             "output_token_details": {"text_tokens": 44, "audio_tokens": 169},
# #         },
# #         "metadata": None,
# #     },
# # }

# # call_transcripts: Dict[str, List[dict]] = {}  # session_id -> list of conversation turns

# # call_transcripts['1'] = []

# # if response.get("type") == "response.done":

# #     inner_response = response.get("response", {})
# #     output_list = inner_response.get("output", [])

# #     if output_list:
# #         first_output_item = output_list[0]
# #         print(first_output_item)
# #         content_list = first_output_item.get("content", [])

# #         if content_list:
# #             first_content_item = content_list[0]
# #             transcript = first_content_item.get("transcript")
# #             print(transcript)

# #             if transcript:
# #                 call_transcripts['1'].append({
# #                     "role": "assistant",
# #                     "content": transcript
# #                 })
# #                 print(call_transcripts['1'], "&&&&&&&&&&&&&&&&&&&&&&&&&")

# # transcribe audio in user conversation,
# import asyncio
# # import re
# # import time
# # import json
# # import requests
# from demo import get_payment_status, send_payment_link
# # from utils.logger import get_logger
# # from config.load_config import Config
# # from utils.transcribe_call_record import transcribe_user_call_record
# # from utils.rewrite_call_record import rewrite_call_record_with_user_conversation
# # config = Config()
# # logger = get_logger()

# # def transcribe_user_call_record(filepath):
# #     try:
# #         start_time = time.time()
# #         # Open the written file for reading as binary
# #         with open(filepath, "rb") as audio_file:
# #             headers = {"Authorization": f"Bearer {config.GROQ_API_KEY}"}
# #             files = {"file": ("audio.wav", audio_file, "audio/wav")}

# #             response = requests.post(
# #                 "https://api.groq.com/openai/v1/audio/transcriptions",
# #                 headers=headers,
# #                 files=files,
# #                 data={"model": "whisper-large-v3"},
# #                 timeout=10
# #             )

# #         if response.status_code == 200:
# #             transcribed_text = response.json().get("text", "")
# #             duration = time.time() - start_time
# #             logger.info(f"TRANSCRIBED TEXT in {duration:.2f}s: {transcribed_text}")
# #             return transcribed_text.split()
# #         else:
# #             logger.error(f"Transcription API error: {response.status_code} {response.text}")
# #             return ""
# #     except requests.exceptions.Timeout:
# #         logger.error("Transcription request timed out")
# #         return ""
# #     except requests.exceptions.RequestException as e:
# #         logger.error(f"Transcription request error: {e}")
# #         return ""
# #     except Exception as e:
# #         logger.error(f"Transcription processing error: {e}")
# #         return ""

# # # pass call record and user conversation then use llm to rewrite the call record append user conversation to call record
# # def rewrite_call_record_with_user_conversation(call_record_json, user_sentences):
# #     try:
# #         start_time = time.time()
# #         assistant_transcript = call_record_json.get("transcript", [])

# #         # Create a properly ordered conversation with user speaking first
# #         combined_conversation = []
# #         # First user message starts the conversation
# #         if user_sentences:
# #             combined_conversation.append({
# #                 "role": "user",
# #                 "content": user_sentences[0].strip()
# #             })
# #         # Then properly interleave assistant and remaining user messages
# #         for i in range(len(assistant_transcript)):
# #             # Add assistant message
# #             combined_conversation.append(assistant_transcript[i])
# #             # Add user response if available (starting from index 1 since we used index 0 above)
# #             if i + 1 < len(user_sentences):
# #                 combined_conversation.append({
# #                     "role": "user",
# #                     "content": user_sentences[i + 1].strip()
# #                 })
# #         # Prompt to generate clean summary, tags, and transcript based on complete conversation
# #         prompt = (
# #             "You are an AI assistant helping clean up and rewrite customer support call records.\n"
# #             "Given the full transcript (assistant and user), generate a clean, structured call record.\n"
# #             "Your response must be a valid JSON with the following format:\n\n"
# #             "{\n"
# #             '  "summary": "<concise summary of the call>",\n'
# #             '  "tags": ["<tag1>", "<tag2>", ...],\n'
# #             '  "transcript": [\n'
# #             '    {"role": "user", "content": "..."},\n'
# #             '    {"role": "assistant", "content": "..."},\n'
# #             "    ...\n"
# #             "  ]\n"
# #             "}\n\n"
# #             "IMPORTANT: The conversation MUST follow this pattern: user speaks first, then assistant responds, then user speaks again, etc. "
# #             "Ensure the summary captures the key actions (e.g. payment made, information confirmed), and tags reflect the conversation outcome."
# #         )

# #         messages = [{"role": "system", "content": prompt}]
# #         # Add the combined conversation as user message for context
# #         messages.append({
# #             "role": "user",
# #             "content": "Here is the conversation transcript. Make sure to keep user speaking first, then assistant, then user, etc.:\n" +
# #                       json.dumps(combined_conversation, indent=2)
# #         })

# #         headers = {
# #             "Authorization": f"Bearer {config.GROQ_API_KEY}",
# #             "Content-Type": "application/json",
# #         }

# #         payload = {
# #             "model": "llama3-8b-8192",
# #             "messages": messages,
# #             "temperature": 0.5,  # Lower temperature for more consistent formatting
# #             "max_tokens": 1500,
# #         }

# #         response = requests.post(
# #             "https://api.groq.com/openai/v1/chat/completions",
# #             headers=headers,
# #             json=payload,
# #             timeout=15
# #         )

# #         if response.status_code == 200:
# #             llm_response = response.json()["choices"][0]["message"]["content"]
# #             duration = time.time() - start_time
# #             logger.info(f"Response generated in {duration:.2f}s")
# #             # Parse back to JSON
# #             try:
# #                 rewritten_json = json.loads(llm_response)
# #                 # Validate transcript order (user first, then assistant)
# #                 transcript = rewritten_json.get("transcript", [])
# #                 if transcript and transcript[0].get("role") != "user":
# #                     # Fix the order if not correct
# #                     corrected_transcript = []
# #                     for i in range(len(user_sentences)):
# #                         corrected_transcript.append({
# #                             "role": "user",
# #                             "content": user_sentences[i].strip()
# #                         })
# #                         if i < len(assistant_transcript):
# #                             corrected_transcript.append(assistant_transcript[i])
# #                     rewritten_json["transcript"] = corrected_transcript
# #                 return rewritten_json
# #             except json.JSONDecodeError:
# #                 logger.error("Failed to parse LLM response as JSON.")
# #                 # Try to extract JSON if it's embedded in other text
# #                 match = re.search(r'({[\s\S]*})', llm_response)
# #                 if match:
# #                     try:
# #                         rewritten_json = json.loads(match.group(1))
# #                         return rewritten_json
# #                     except:
# #                         pass
# #                 return {"error": "Failed to parse LLM output.", "raw_output": llm_response}
# #         else:
# #             logger.error(f"LLM API error: {response.status_code} {response.text}")
# #             return {"error": f"LLM API error: {response.status_code}"}
# #     except requests.exceptions.Timeout:
# #         logger.error("LLM request timed out")
# #         return {"error": "Request timed out"}
# #     except requests.exceptions.RequestException as e:
# #         logger.error(f"LLM request error: {e}")
# #         return {"error": f"Request exception: {str(e)}"}
# #     except Exception as e:
# #         logger.error(f"LLM generation error: {e}")
# #         return {"error": str(e)}

# # Your async workflow wrapped in a coroutine
# # async def main():
# #     filepath = "recordings/619188cd-8fde-4ab2-a6e8-69f9760f84e2.wav"
# #     user_sentences = await transcribe_user_call_record(filepath)
# #     print("User Sentences:", user_sentences)

# #     call_record_path = "call_records/f96c2375-585b-41a9-af10-6fe8c74ec53a.json"
# #     rewritten = await rewrite_call_record_with_user_conversation(call_record_path, user_sentences)

# #     print(json.dumps(rewritten, indent=2))

import asyncio
from demo import send_payment_link


async def main():
    print("::::::::::::::::::::::::::")
    # payment_status = await get_payment_status(call_id="call_17FSzyr47AqHycg3")
    await send_payment_link(phone_number="+917011897710",amount=1245,call_id="jl;akdjs")
    # print(payment_status)

# Entry point
if __name__ == "__main__":
    asyncio.run(main())


# # Optional: Save to file
# with open("call_records/rewritten_call_record.json", "w") as f:
#     json.dump(rewritten, f, indent=2)
