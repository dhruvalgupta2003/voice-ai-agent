def validate_config(config):
    """Validate all required environment/config variables."""
    missing = []
    if not config.OPENAI_API_KEY:
        raise ValueError('Missing the OpenAI API key. Please set it in the .env file.')

    if not config.GROQ_API_KEY:
        missing.append("Groq API key")

    if not config.AWS_ACCESS_KEY or not config.AWS_SECRET_KEY:
        missing.append("AWS credentials")

    if (
        not config.TWILIO_ACCOUNT_SID
        or not config.TWILIO_AUTH_TOKEN
        or not config.TWILIO_PHONE_NUMBER
    ):
        missing.append("Twilio configuration")

    if missing:
        raise ValueError(f"Missing configuration: {', '.join(missing)}. Please set them in the .env file.")
