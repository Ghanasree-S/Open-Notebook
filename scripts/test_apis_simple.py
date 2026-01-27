"""
Simple standalone test for Sarvam + Groq APIs
No project dependencies required - just httpx
"""

import asyncio
import base64
import os
import sys
from pathlib import Path

# Load .env manually
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ[key.strip()] = value.strip()

# Try to import httpx, install if not available
try:
    import httpx
except ImportError:
    print("Installing httpx...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "httpx"], check=True)
    import httpx


async def test_groq():
    """Test Groq API"""
    print("\n[1/2] Testing Groq LLM...")
    
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key or "your_" in api_key:
        print("    FAILED: GROQ_API_KEY not set in .env file")
        print("    Get your key at: https://console.groq.com/keys")
        return False
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": [
                        {"role": "user", "content": "Say 'Hello' in one word."}
                    ],
                    "max_tokens": 10,
                },
            )
            
            if response.status_code == 200:
                result = response.json()
                message = result["choices"][0]["message"]["content"]
                print(f"    SUCCESS: Groq responded: {message}")
                return True
            else:
                print(f"    FAILED: Status {response.status_code}")
                print(f"    Response: {response.text[:200]}")
                return False
    except Exception as e:
        print(f"    FAILED: {e}")
        return False


async def test_sarvam():
    """Test Sarvam TTS API"""
    print("\n[2/2] Testing Sarvam TTS...")
    
    api_key = os.environ.get("SARVAM_API_KEY", "")
    if not api_key or "your_" in api_key:
        print("    FAILED: SARVAM_API_KEY not set in .env file")
        print("    Get your key at: https://dashboard.sarvam.ai")
        return False
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api.sarvam.ai/text-to-speech",
                headers={
                    "Content-Type": "application/json",
                    "api-subscription-key": api_key,
                },
                json={
                    "inputs": ["Hello, this is a test."],
                    "target_language_code": "en-IN",
                    "speaker": "anushka",
                    "model": "bulbul:v2",
                    "pace": 1.0,
                    "sample_rate": 22050,
                    "enable_preprocessing": True,
                },
            )
            
            if response.status_code == 200:
                result = response.json()
                if "audios" in result and result["audios"]:
                    audio_base64 = result["audios"][0]
                    audio_bytes = base64.b64decode(audio_base64)
                    
                    # Save test audio
                    output_path = Path(__file__).parent.parent / "data" / "test_sarvam.wav"
                    output_path.parent.mkdir(exist_ok=True)
                    with open(output_path, "wb") as f:
                        f.write(audio_bytes)
                    
                    print(f"    SUCCESS: Audio generated ({len(audio_bytes)} bytes)")
                    print(f"    Saved to: {output_path}")
                    return True
                else:
                    print("    FAILED: No audio returned")
                    return False
            else:
                print(f"    FAILED: Status {response.status_code}")
                print(f"    Response: {response.text[:200]}")
                return False
    except Exception as e:
        print(f"    FAILED: {e}")
        return False


async def main():
    print("=" * 50)
    print("  GROQ + SARVAM API TEST")
    print("=" * 50)
    
    groq_ok = await test_groq()
    sarvam_ok = await test_sarvam()
    
    print("\n" + "=" * 50)
    print("  RESULTS")
    print("=" * 50)
    print(f"  Groq LLM:    {'PASSED' if groq_ok else 'FAILED'}")
    print(f"  Sarvam TTS:  {'PASSED' if sarvam_ok else 'FAILED'}")
    print("=" * 50)
    
    if groq_ok and sarvam_ok:
        print("\nBoth APIs are working! Your podcast integration is ready.")
        print("\nNext steps:")
        print("1. Start your Open-Notebook application")
        print("2. Configure speaker profiles with Sarvam voice IDs")
        print("3. Generate podcasts!")
    else:
        print("\nPlease fix the failing tests above.")
        print("Make sure your .env file has valid API keys.")


if __name__ == "__main__":
    asyncio.run(main())
