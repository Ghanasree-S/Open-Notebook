"""
Reset models - Remove all models and add only the required ones for podcasts.
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


async def reset_models():
    """Remove all models and add only required ones"""
    
    from open_notebook.database.repository import repo_query
    
    print("=" * 50)
    print("  Resetting Models Database")
    print("=" * 50)
    
    # Step 1: Delete all existing models
    print("\n[1] Removing all existing models...")
    try:
        await repo_query("DELETE model")
        print("    All models removed.")
    except Exception as e:
        print(f"    Error removing models: {e}")
    
    # Step 2: Add only the models we need
    print("\n[2] Adding required models...")
    
    models = [
        # Groq LLM - for podcast script generation
        {
            "name": "llama-3.3-70b-versatile",
            "provider": "groq",
            "type": "language",
        },
        # Sarvam TTS - Indian language text-to-speech
        {
            "name": "bulbul:v2",
            "provider": "sarvam",
            "type": "text_to_speech",
        },
        # Groq STT - speech-to-text (Whisper)
        {
            "name": "whisper-large-v3",
            "provider": "groq",
            "type": "speech_to_text",
        },
    ]
    
    for model in models:
        try:
            await repo_query(
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
            print(f"    [ADDED] {model['provider']}/{model['name']} ({model['type']})")
        except Exception as e:
            print(f"    [ERROR] {model['provider']}/{model['name']}: {e}")
    
    print("\n" + "=" * 50)
    print("  Done! Models reset to minimal set.")
    print("=" * 50)
    print("\nAvailable models:")
    print("  - groq/llama-3.3-70b-versatile (LLM)")
    print("  - sarvam/bulbul:v2 (TTS)")
    print("  - groq/whisper-large-v3 (STT)")
    print("\nRefresh your browser to see the changes!")


if __name__ == "__main__":
    asyncio.run(reset_models())
