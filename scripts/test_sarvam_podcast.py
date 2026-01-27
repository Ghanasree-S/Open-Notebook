"""
Test script for Sarvam + Groq Podcast Generation

This script tests the Sarvam TTS and Groq LLM integration for podcast generation.

Usage:
    1. Set your API keys in .env file:
       - GROQ_API_KEY=your_groq_key
       - SARVAM_API_KEY=your_sarvam_key
    
    2. Run this script:
       python scripts/test_sarvam_podcast.py
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv not installed, try to load .env manually
    import os
    env_path = project_root / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key.strip()] = value.strip()


async def test_sarvam_tts():
    """Test Sarvam TTS alone"""
    print("=" * 60)
    print("Testing Sarvam TTS...")
    print("=" * 60)
    
    try:
        from open_notebook.services.sarvam_tts import SarvamTTSProvider, SarvamTTSConfig
        
        api_key = os.environ.get("SARVAM_API_KEY")
        if not api_key or api_key == "your_sarvam_api_key_here":
            print("❌ SARVAM_API_KEY not set. Please add it to your .env file.")
            print("   Get your key at: https://dashboard.sarvam.ai")
            return False
        
        config = SarvamTTSConfig(api_key=api_key)
        provider = SarvamTTSProvider(config)
        
        # Test with a simple text
        test_text = "Hello! This is a test of the Sarvam AI text to speech system."
        output_path = str(project_root / "data" / "test_sarvam_output.wav")
        
        print(f"Converting text: '{test_text}'")
        print(f"Using voice: Anushka")
        
        result = await provider.synthesize(
            text=test_text,
            voice_id="Anushka",
            output_path=output_path,
        )
        
        if Path(result).exists():
            file_size = Path(result).stat().st_size
            print(f"✅ Sarvam TTS test PASSED!")
            print(f"   Audio file saved to: {result}")
            print(f"   File size: {file_size} bytes")
            return True
        else:
            print("❌ Sarvam TTS test FAILED - file not created")
            return False
            
    except Exception as e:
        print(f"❌ Sarvam TTS test FAILED: {e}")
        return False


async def test_groq_llm():
    """Test Groq LLM alone"""
    print("\n" + "=" * 60)
    print("Testing Groq LLM...")
    print("=" * 60)
    
    try:
        import httpx
        
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key or api_key == "your_groq_api_key_here":
            print("❌ GROQ_API_KEY not set. Please add it to your .env file.")
            print("   Get your key at: https://console.groq.com/keys")
            return False
        
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
                        {"role": "user", "content": "Say 'Hello, Groq is working!' in one sentence."}
                    ],
                    "max_tokens": 50,
                },
            )
            
            if response.status_code == 200:
                result = response.json()
                message = result["choices"][0]["message"]["content"]
                print(f"✅ Groq LLM test PASSED!")
                print(f"   Response: {message}")
                return True
            else:
                print(f"❌ Groq LLM test FAILED: {response.status_code} - {response.text}")
                return False
                
    except Exception as e:
        print(f"❌ Groq LLM test FAILED: {e}")
        return False


async def test_full_podcast():
    """Test full podcast generation"""
    print("\n" + "=" * 60)
    print("Testing Full Podcast Generation (Groq + Sarvam)...")
    print("=" * 60)
    
    try:
        from open_notebook.services.sarvam_podcast import generate_podcast_with_sarvam
        
        # Check API keys
        groq_key = os.environ.get("GROQ_API_KEY")
        sarvam_key = os.environ.get("SARVAM_API_KEY")
        
        if not groq_key or groq_key == "your_groq_api_key_here":
            print("❌ GROQ_API_KEY not set")
            return False
        
        if not sarvam_key or sarvam_key == "your_sarvam_api_key_here":
            print("❌ SARVAM_API_KEY not set")
            return False
        
        # Test content
        test_content = """
        Artificial Intelligence is transforming the world in remarkable ways.
        From healthcare to transportation, AI is making processes more efficient
        and creating new possibilities. Machine learning algorithms can now
        diagnose diseases, drive cars, and even create art.
        """
        
        output_dir = str(project_root / "data" / "test_podcast")
        
        print("Generating test podcast...")
        print("This may take a few minutes...")
        
        result = await generate_podcast_with_sarvam(
            content=test_content,
            episode_name="test_episode",
            speaker1_name="Priya",
            speaker1_voice="Anushka",
            speaker2_name="Raj",
            speaker2_voice="Abhilash",
            briefing="Create a short 3-segment conversation about AI.",
            output_dir=output_dir,
        )
        
        if result.get("final_output_path") and Path(result["final_output_path"]).exists():
            print(f"✅ Full podcast generation PASSED!")
            print(f"   Audio file: {result['final_output_path']}")
            print(f"   Transcript: {result.get('transcript_path')}")
            return True
        else:
            print("❌ Full podcast generation FAILED - audio file not created")
            return False
            
    except Exception as e:
        print(f"❌ Full podcast generation FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    print("\n" + "=" * 60)
    print("  SARVAM + GROQ PODCAST INTEGRATION TEST")
    print("=" * 60)
    
    # Create data directory
    data_dir = project_root / "data"
    data_dir.mkdir(exist_ok=True)
    
    # Run tests
    tts_ok = await test_sarvam_tts()
    llm_ok = await test_groq_llm()
    
    if tts_ok and llm_ok:
        podcast_ok = await test_full_podcast()
    else:
        print("\n⚠️ Skipping full podcast test due to earlier failures")
        podcast_ok = False
    
    # Summary
    print("\n" + "=" * 60)
    print("  TEST SUMMARY")
    print("=" * 60)
    print(f"  Sarvam TTS:     {'✅ PASSED' if tts_ok else '❌ FAILED'}")
    print(f"  Groq LLM:       {'✅ PASSED' if llm_ok else '❌ FAILED'}")
    print(f"  Full Podcast:   {'✅ PASSED' if podcast_ok else '❌ FAILED'}")
    print("=" * 60)
    
    if tts_ok and llm_ok and podcast_ok:
        print("\n🎉 All tests passed! Your Sarvam + Groq integration is ready!")
        print("\nNext steps:")
        print("1. Create speaker profiles in the UI with Sarvam voice IDs")
        print("2. Create episode profiles with Groq models")
        print("3. Generate podcasts!")
    elif not tts_ok or not llm_ok:
        print("\n⚠️ Please fix the failing tests above before proceeding.")
        print("\nMake sure you have set the API keys in your .env file:")
        print("  GROQ_API_KEY=your_actual_groq_key")
        print("  SARVAM_API_KEY=your_actual_sarvam_key")


if __name__ == "__main__":
    asyncio.run(main())
