import time
import json
import requests
from utils.logger import get_logger
from config.load_config import Config

config = Config()
logger = get_logger()

def extract_json_from_response(response_text):
    try:
        start = response_text.find("{")
        end = response_text.rfind("}") + 1
        if start != -1 and end != -1:
            json_str = response_text[start:end]
            return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error during extraction: {e}")
    return None

async def rewrite_call_record_with_user_conversation(call_record_path, user_sentences):
    try:
        start_time = time.time()
        
        # Read the call record from the provided file path
        with open(call_record_path, "r") as f:
            call_record_json = json.load(f)
        
        # Save original metadata to preserve
        session_id = call_record_json.get("session_id", "")
        call_id = call_record_json.get("call_id", "")
        timestamp = call_record_json.get("timestamp", "")
        original_tags = call_record_json.get("tags", [])
        
        assistant_transcript = call_record_json.get("transcript", [])

        # Create a properly ordered conversation with user speaking first
        combined_conversation = []
        # First user message starts the conversation
        if user_sentences:
            combined_conversation.append({
                "role": "user",
                "content": user_sentences[0].strip()
            })
        # Then properly interleave assistant and remaining user messages
        for i in range(len(assistant_transcript)):
            # Add assistant message
            combined_conversation.append(assistant_transcript[i])
            # Add user response if available (starting from index 1 since we used index 0 above)
            if i + 1 < len(user_sentences):
                combined_conversation.append({
                    "role": "user",
                    "content": user_sentences[i + 1].strip()
                })
        
        # Prompt to generate clean summary, tags, and transcript based on complete conversation
       # Prompt to generate clean summary, tags, and transcript based on complete conversation
        prompt = (
            "You are an AI assistant helping clean up and rewrite customer support call records.\n"
            "Given the full transcript (assistant and user), generate a clean, structured call record.\n"
            "Please analyze this conversation and provide:"
            "1. A concise summary of the call (2-3 sentences)"
            "2. Appropriate tags from this list:"
            "- payment_completed: Customer made a payment during the call"
            "- payment_scheduled: Customer agreed to pay at a later date"
            "- requires_followup: Call needs further follow-up"
            "- financial_hardship: Customer reported financial difficulties"
            "- disputed_loan: Customer disputed loan details or refused payment"
            "- wrong_number: Called wrong person/number"
            "- unsuccessful: Call didn't achieve a resolution"
            "- high_satisfaction: Customer expressed high satisfaction (4-5 rating)"
            "- low_satisfaction: Customer expressed low satisfaction (1-2 rating)"
            "Your response must be a valid JSON with the following format:\n\n"
            "{\n"
            '  "summary": "<concise summary of the call>",\n'
            '  "tags": ["<tag1>", "<tag2>", ...],\n'
            '  "transcript": [\n'
            '    {"role": "user", "content": "..."},\n'
            '    {"role": "assistant", "content": "..."},\n'
            "    ...\n"
            "  ]\n"
            "}\n\n"
            "Don't remove any unnecessary information from the original transcript and focus on the key points discussed during the call just correct and remodel it."
            "Remove duplicate user messages if any, and correct the word and grammer"
            "IMPORTANT: The conversation MUST follow this pattern: user speaks first, then assistant responds, then user speaks again, etc. "
            "Ensure the summary captures the key actions (e.g. payment made, information confirmed), and tags reflect the conversation outcome."
        )

        messages = [{"role": "system", "content": prompt}]
        # Add the combined conversation as user message for context
        messages.append({
            "role": "user",
            "content": "Here is the conversation transcript. Make sure to keep user speaking first, then assistant, then user, etc.:\n" +
                      json.dumps(combined_conversation, indent=2)
        })

        headers = {
            "Authorization": f"Bearer {config.GROQ_API_KEY}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": "llama3-70b-8192",
            "messages": messages,
            "temperature": 0.5,
            "max_tokens": 1500,
        }

        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=15
        )

        if response.status_code != 200:
            logger.error(f"LLM API failed: {response.status_code}, {response.text}")
            return {"error": "Failed to get LLM response", "status_code": response.status_code}
        
        llm_response = response.json()["choices"][0]["message"]["content"]
        duration = time.time() - start_time
        logger.info(f"Response generated in {duration:.2f}s")
        logger.info(f"Rewriting the call record json for call id {call_id}")
        logger.info(f'Rewritten response : {llm_response}')
        # Parse back to JSON
        llm_json = None
        try:
            try:
                # First try to parse the entire response
                llm_json = json.loads(llm_response)
            except json.JSONDecodeError:
                logger.warning("Direct JSON parsing failed. Attempting extraction...")
                llm_json = extract_json_from_response(llm_response)

            if not llm_json:
                logger.error("Failed to extract or parse JSON from LLM response.")
                return {"error": "LLM response is not valid JSON", "raw_output": llm_response}
                
            # Validate transcript order (user first, then assistant)
            transcript = llm_json.get("transcript", [])
            if transcript and transcript[0].get("role") != "user":
                # Fix the order if not correct
                corrected_transcript = []
                for i in range(len(user_sentences)):
                    corrected_transcript.append({
                        "role": "user",
                        "content": user_sentences[i].strip()
                    })
                    if i < len(assistant_transcript):
                        corrected_transcript.append(assistant_transcript[i])
                llm_json["transcript"] = corrected_transcript
            
            # If no tags were generated, use the original tags
            if "tags" not in llm_json or not llm_json["tags"]:
                llm_json["tags"] = original_tags
            
            # Create a new ordered dictionary with fields in the desired order
            rewritten_json = {}
            
            # First metadata fields
            rewritten_json["session_id"] = session_id
            if call_id:  # Only add if it exists
                rewritten_json["call_id"] = call_id
            rewritten_json["timestamp"] = timestamp
            
            # Then content fields
            rewritten_json["summary"] = llm_json.get("summary", "")
            rewritten_json["tags"] = llm_json.get("tags", original_tags)
            rewritten_json["transcript"] = llm_json.get("transcript", [])
            
            # Write the ordered JSON back to the original file
            with open(call_record_path, "w") as f:
                json.dump(rewritten_json, f, indent=2)
            
            return rewritten_json
        
        except json.JSONDecodeError:
            logger.error("Failed to parse LLM response as JSON.")
            # Try to extract JSON if it's embedded in other text
            try:
                # Clean up the response to help with parsing
                cleaned_response = llm_response.strip()
                # Find the first { and last } to extract the JSON object
                start_idx = cleaned_response.find('{')
                end_idx = cleaned_response.rfind('}') + 1
                
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = cleaned_response[start_idx:end_idx]
                    llm_json = json.loads(json_str)
                    
                    # If no tags were generated, use the original tags
                    if "tags" not in llm_json or not llm_json["tags"]:
                        llm_json["tags"] = original_tags
                    
                    # Create a new ordered dictionary with fields in the desired order
                    rewritten_json = {}
                    
                    # First metadata fields
                    rewritten_json["session_id"] = session_id
                    if call_id:  # Only add if it exists
                        rewritten_json["call_id"] = call_id
                    rewritten_json["timestamp"] = timestamp
                    
                    # Then content fields
                    rewritten_json["summary"] = llm_json.get("summary", "")
                    rewritten_json["tags"] = llm_json.get("tags", original_tags)
                    rewritten_json["transcript"] = llm_json.get("transcript", [])
                    
                    # Write the ordered JSON back to the original file
                    with open(call_record_path, "w") as f:
                        json.dump(rewritten_json, f, indent=2)
                        
                    return rewritten_json
                else:
                    raise ValueError("Could not find valid JSON in response")
            except Exception as e:
                logger.error(f"Error extracting JSON: {e}")
                logger.error(f"Raw response: {llm_response}")
                return {"error": "Failed to parse LLM output.", "raw_output": llm_response}
    
        else:
            logger.error(f"LLM API error: {response.status_code} {response.text}")
            return {"error": f"LLM API error: {response.status_code}"}
    
    except requests.exceptions.Timeout:
        logger.error("LLM request timed out")
        return {"error": "Request timed out"}
    except requests.exceptions.RequestException as e:
        logger.error(f"LLM request error: {e}")
        return {"error": f"Request exception: {str(e)}"}
    except FileNotFoundError:
        logger.error(f"Call record file not found: {call_record_path}")
        return {"error": f"File not found: {call_record_path}"}
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing call record file: {e}")
        return {"error": f"Invalid JSON in call record file: {str(e)}"}
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return {"error": str(e)}