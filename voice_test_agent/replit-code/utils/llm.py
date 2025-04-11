import os
import logging
import asyncio
from typing import Optional
from groq import Groq, AsyncGroq

logger = logging.getLogger(__name__)

# Initialize Groq client

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY)
async_groq_client = AsyncGroq(api_key=GROQ_API_KEY)


async def get_llm_response(text: str, system_message: str) -> Optional[str]:
    """
    Get response from Groq LLM
    
    Args:
        text: The user's transcribed text
        system_message: The system message to guide the LLM's responses
        
    Returns:
        The LLM's response or None if an error occurred
    """
    try:
        logger.info(f"Sending request to LLM: {text}")

        # Use the async client for better performance
        completion = await async_groq_client.chat.completions.create(
            messages=[{
                "role": "system",
                "content": system_message
            }, {
                "role": "user",
                "content": text
            }],
            model="llama3-70b-8192",
            temperature=0.7,
            max_tokens=256,  # Limit token count for voice responses
        )

        response = completion.choices[0].message.content
        logger.info(f"LLM response: {response}")
        return response
    except Exception as e:
        logger.error(f"Error getting LLM response: {e}", exc_info=True)

        # Fallback to synchronous client if async fails
        try:

            def run_sync_request():
                completion = groq_client.chat.completions.create(
                    messages=[{
                        "role": "system",
                        "content": system_message
                    }, {
                        "role": "user",
                        "content": text
                    }],
                    model="llama3-70b-8192",
                    temperature=0.7,
                    max_tokens=256,
                )
                return completion.choices[0].message.content

            response = await asyncio.get_event_loop().run_in_executor(
                None, run_sync_request)
            logger.info(f"Sync LLM response: {response}")
            return response
        except Exception as fallback_error:
            logger.error(f"Error in fallback LLM response: {fallback_error}",
                         exc_info=True)
            return "I'm sorry, I couldn't process that. Could you please try again?"
