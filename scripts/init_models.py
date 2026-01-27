"""
Initialize default models for Groq + Sarvam/Google TTS podcast setup.
Run this script once to add the required models to the database.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
env_path = project_root / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ[key.strip()] = value.strip()


async def init_models():
    """Add default models to the database"""
    
    # Import after setting up environment
    from open_notebook.database.repository import repo_query
    
    print("=" * 50)
    print("  Initializing Default Models")
    print("=" * 50)
    
    # Models to add
    models = [
        # Groq LLM models
        {
            "name": "llama-3.3-70b-versatile",
            "provider": "groq",
            "type": "language",
        },
        {
            "name": "llama-3.1-8b-instant",
            "provider": "groq", 
            "type": "language",
        },
        # Google TTS
        {
            "name": "gemini-2.5-flash-preview-tts",
            "provider": "google",
            "type": "text_to_speech",
        },
        # Groq STT (Whisper)
        {
            "name": "whisper-large-v3",
            "provider": "groq",
            "type": "speech_to_text",
        },
        # Sarvam TTS models (Indian languages)
        {
            "name": "bulbul:v2",
            "provider": "sarvam",
            "type": "text_to_speech",
        },
        {
            "name": "bulbul:v3-beta",
            "provider": "sarvam",
            "type": "text_to_speech",
        },
    ]
    
    for model in models:
        try:
            # Check if model already exists
            existing = await repo_query(
                "SELECT * FROM model WHERE name = $name AND provider = $provider",
                {"name": model["name"], "provider": model["provider"]}
            )
            
            if existing:
                print(f"  [EXISTS] {model['provider']}/{model['name']}")
                continue
            
            # Create model
            result = await repo_query(
                """
                CREATE model SET
                    name = $name,
                    provider = $provider,
                    type = $type,
                    created = time::now(),
                    updated = time::now()
                """,
                model
            )
            print(f"  [ADDED]  {model['provider']}/{model['name']} ({model['type']})")
            
        except Exception as e:
            print(f"  [ERROR]  {model['provider']}/{model['name']}: {e}")
    
    print("=" * 50)
    print("  Done! Refresh your browser to see the models.")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(init_models())
