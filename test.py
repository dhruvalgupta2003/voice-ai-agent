# from datetime import datetime
# from utils.load_prompt import load_prompt
# from config.load_config import Config

# config = Config()

# SYSTEM_MESSAGE= load_prompt('loan_repayment_prompt')

# data:dict = {
#     "customer_name": "Michael",
#     "account_number": "12345678",
#     "loan_amount": "10000",
#     "due_date": "15th April 2025"
# }

# # Get loan account details - in production this would come from your CRM/database
# customer_name = data.get("customer_name", "Michael")
# account_number = data.get("account_number", "1234")
# loan_amount = data.get("loan_amount", "$10,000")
# due_date = data.get("due_date", "15th April 2025")
# parsed_due_date = datetime.strptime(data.get("due_date", "15th April 2025"), "%dth %B %Y").date()

# # Determine overdue or upcoming
# today = datetime.today().date()
# overdue_or_upcoming = "Overdue" if parsed_due_date < today else "Upcoming"


# # Create customized system prompt with call details
# custom_prompt = SYSTEM_MESSAGE.replace("[Customer Name]", customer_name)
# custom_prompt = custom_prompt.replace("XXXX", account_number[-4:] if len(account_number) >= 4 else account_number)
# custom_prompt = custom_prompt.replace("₹XX,XXX", f"${loan_amount}")
# custom_prompt = custom_prompt.replace("[Date]", due_date)
# custom_prompt = custom_prompt.replace("[Agent Name]", config.AGENT_NAME)
# custom_prompt = custom_prompt.replace("[Company Name]", config.COMPANY_NAME)
# custom_prompt = custom_prompt.replace("[Overdue/Upcoming in X days]", overdue_or_upcoming)

# print(custom_prompt)

from typing import Dict, List


response = {
    "type": "response.done",
    "event_id": "event_BKkyxH7qKdFzWW5gU9o16",
    "response": {
        "object": "realtime.response",
        "id": "resp_BKkyvYrRVCILg4qVAO4BB",
        "status": "completed",
        "status_details": None,
        "output": [
            {
                "id": "item_BKkyvi2BwdT0aMwU141Eg",
                "object": "realtime.item",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "audio",
                        "transcript": "Hello, this is Ava calling on behalf of OakStone-Mortgages regarding your loan account ending with 1234. Is this Michael?",
                    }
                ],
            }
        ],
        "conversation_id": "conv_BKkyuRrDYuCmbDJ4d7tQT",
        "modalities": ["audio", "text"],
        "voice": "alloy",
        "output_audio_format": "g711_ulaw",
        "temperature": 0.8,
        "max_output_tokens": "inf",
        "usage": {
            "total_tokens": 749,
            "input_tokens": 536,
            "output_tokens": 213,
            "input_token_details": {
                "text_tokens": 528,
                "audio_tokens": 8,
                "cached_tokens": 0,
                "cached_tokens_details": {"text_tokens": 0, "audio_tokens": 0},
            },
            "output_token_details": {"text_tokens": 44, "audio_tokens": 169},
        },
        "metadata": None,
    },
}

call_transcripts: Dict[str, List[dict]] = {}  # session_id -> list of conversation turns

call_transcripts['1'] = []

if response.get("type") == "response.done":

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

            if transcript:
                call_transcripts['1'].append({
                    "role": "assistant",
                    "content": transcript
                })
                print(call_transcripts['1'], "&&&&&&&&&&&&&&&&&&&&&&&&&")
