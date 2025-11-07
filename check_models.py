from groq import Groq
import os

# Reads Groq API key from env: set GROQ_API_KEY or use Streamlit secrets when running the app
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    print("GROQ_API_KEY is not set in environment variables.")
else:
    try:
        client = Groq(api_key=api_key)
        print("Fetching available Groq models...\n")
        models = client.models.list()
        found_any = False
        for m in models.data:
            print(f"Model: {m.id}")
            found_any = True
        if not found_any:
            print("No models returned for the provided API key.")
    except Exception as e:
        print(f"An error occurred: {e}")