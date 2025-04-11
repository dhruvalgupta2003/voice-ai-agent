import os
from typing import List, Tuple
import json
import httpx
from enum import Enum
from utils.logger import get_logger

logger = get_logger()

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

async def generate_call_summary_and_tags(transcript: List[dict]) -> Tuple[str, List[CallOutcomeTag]]:
    """
    Generate call summary and tags using Groq LLM API.
    
    Args:
        transcript: List of conversation turns with 'role' and 'content' keys
        
    Returns:
        Tuple containing summary string and list of CallOutcomeTags
    """
    # Convert transcript to a readable conversation format
    conversation_text = "\n".join([
        f"{'Agent' if item['role'] == 'assistant' else 'Customer'}: {item['content']}" 
        for item in transcript if 'content' in item
    ])
    
    # Prepare Groq API request
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        # Fallback to simple rule-based approach if API key not available
        return fallback_summary_generation(conversation_text)
    
    # Prepare the prompt for Groq
    prompt = f"""
    Below is a transcript of a conversation between a loan collection agent and a customer.
    Please analyze this conversation and provide:
    1. A concise summary of the call (2-3 sentences)
    2. Appropriate tags from this list:
       - payment_completed: Customer made a payment during the call
       - payment_scheduled: Customer agreed to pay at a later date
       - requires_followup: Call needs further follow-up
       - financial_hardship: Customer reported financial difficulties
       - disputed_loan: Customer disputed loan details or refused payment
       - wrong_number: Called wrong person/number
       - unsuccessful: Call didn't achieve a resolution
       - high_satisfaction: Customer expressed high satisfaction (4-5 rating)
       - low_satisfaction: Customer expressed low satisfaction (1-2 rating)
    
    Format your response as a JSON object with "summary" and "tags" fields.
    
    Conversation transcript:
    {conversation_text}
    """
    
    try:
        logger.info('Generating call summary and tags ...')
        # Call Groq API
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {groq_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "llama3-70b-8192",  # Using LLaMa3-70B as it's one of Groq's fastest models
                    "messages": [
                        {"role": "system", "content": "You are an AI assistant that analyzes call transcripts and provides summaries and tags."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.2,  # Lower temperature for more consistent results
                    "response_format": {"type": "json_object"}
                }
            )
            
            if response.status_code != 200:
                print(f"Error from Groq API: {response.status_code} {response.text}")
                return fallback_summary_generation(conversation_text)
            
            result = response.json()
            llm_response = result["choices"][0]["message"]["content"]
            
            # Parse the JSON response
            try:
                parsed_response = json.loads(llm_response)
                summary = parsed_response.get("summary", "Call completed.")
                tag_strings = parsed_response.get("tags", ["requires_followup"])
                
                # Convert tag strings to enum values
                tags = []
                for tag_str in tag_strings:
                    try:
                        tag = CallOutcomeTag(tag_str)
                        tags.append(tag)
                    except ValueError:
                        print(f"Invalid tag received from LLM: {tag_str}")
                
                # Add default tag if none were valid
                if not tags:
                    tags.append(CallOutcomeTag.REQUIRES_FOLLOWUP)
                
                return summary, tags
                
            except json.JSONDecodeError:
                print(f"Failed to parse LLM response: {llm_response}")
                return fallback_summary_generation(conversation_text)
                
    except Exception as e:
        print(f"Error using Groq LLM: {str(e)}")
        return fallback_summary_generation(conversation_text)

def fallback_summary_generation(conversation_text: str) -> Tuple[str, List[CallOutcomeTag]]:
    """
    Fallback method for generating summary and tags when Groq API is unavailable.
    Uses simple rule-based approach.
    
    Args:
        conversation_text: String representation of the conversation
        
    Returns:
        Tuple containing summary string and list of CallOutcomeTags
    """
    summary = "Call summary: "
    tags = []
    
    # Check for key phrases to determine outcome
    if "wrong person" in conversation_text.lower() or "wrong number" in conversation_text.lower():
        summary += "Called wrong number."
        tags.append(CallOutcomeTag.WRONG_NUMBER)
    
    elif "payment is confirmed" in conversation_text.lower() or "payment successful" in conversation_text.lower():
        summary += "Customer made payment during the call."
        tags.append(CallOutcomeTag.PAYMENT_COMPLETED)
    
    elif "good time for you to make the payment" in conversation_text.lower() and "reminder" in conversation_text.lower():
        summary += "Customer scheduled payment for later date."
        tags.append(CallOutcomeTag.PAYMENT_SCHEDULED)
    
    elif "financial hardship" in conversation_text.lower() or "can't pay" in conversation_text.lower():
        summary += "Customer reported financial difficulties."
        tags.append(CallOutcomeTag.FINANCIAL_HARDSHIP)
    
    elif "dispute" in conversation_text.lower() or "refuses to pay" in conversation_text.lower():
        summary += "Customer disputed loan details or refused payment."
        tags.append(CallOutcomeTag.DISPUTED_LOAN)
    
    # Check for satisfaction level
    if "satisfied" in conversation_text.lower() and ("4" in conversation_text or "5" in conversation_text):
        summary += " Customer expressed high satisfaction."
        tags.append(CallOutcomeTag.HIGH_SATISFACTION)
    elif "satisfied" in conversation_text.lower() and ("1" in conversation_text or "2" in conversation_text):
        summary += " Customer expressed low satisfaction."
        tags.append(CallOutcomeTag.LOW_SATISFACTION)
    
    # Default if no patterns matched
    if not tags:
        summary += "Call completed without clear resolution."
        tags.append(CallOutcomeTag.REQUIRES_FOLLOWUP)
    
    return summary, tags


if __name__ == "__main__":
    
    async def analyze_transcript():
        transcript = [
            {"role": "assistant", "content": "Hi Michael, this is Jenny from FutureTrust Mortgage. I'm calling about your loan repayment due on 15th April."},
            {"role": "user", "content": "Okay, yes, I remember that."},
            {"role": "assistant", "content": "Would this be a good time to make the payment? I can send you a secure link."},
            {"role": "user", "content": "Sure, send it over."},
            {"role": "assistant", "content": "Payment link has been sent to your phone via SMS. Please let me know once done."},
            {"role": "user", "content": "Done. I’ve made the payment. Thank you."},
            {"role": "assistant", "content": "Thank you for confirming. On a scale of 1-5, how satisfied are you with this call?"},
            {"role": "user", "content": "I’d say 5. Great help!"}
        ]

        summary, tags = await generate_call_summary_and_tags(transcript)
        print("Call Summary:", summary)
        print("Tags:", [tag.value for tag in tags])

    import asyncio
    asyncio.run(analyze_transcript())