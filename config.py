import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # Server Configuration
    APP_NAME = os.getenv("APP_NAME", "Gym AI Service")
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8080"))

settings = Settings()