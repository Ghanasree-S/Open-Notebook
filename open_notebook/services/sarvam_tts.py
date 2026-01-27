"""
Sarvam AI TTS Provider for Open Notebook Podcasts

This module provides integration with Sarvam AI's Text-to-Speech API,
specifically designed for Indian languages with support for:
- bulbul:v2 (default): Supports pitch, loudness, and pace controls
- bulbul:v3-beta: Newer model with temperature control and improved quality

Sarvam TTS Voices:
- bulbul:v2: Anushka, Manisha, Vidya, Arya (Female) | Abhilash, Karun, Hitesh (Male)
- bulbul:v3-beta: Aditya, Ritu, Priya, Neha, Rahul, Pooja, Rohan, Simran, Kavya, 
                  Amit, Dev, Ishita, Shreya, Ratan, Varun, Manan, Sumit, Roopa, 
                  Kabir, Aayan, Shubh, Ashutosh, Advait, Amelia, Sophia
"""

import base64
import os
from pathlib import Path
from typing import Literal, Optional

import httpx
from loguru import logger
from pydantic import BaseModel, Field


class SarvamTTSConfig(BaseModel):
    """Configuration for Sarvam TTS API"""
    
    api_key: str = Field(default_factory=lambda: os.environ.get("SARVAM_API_KEY", ""))
    model: Literal["bulbul:v2", "bulbul:v3-beta"] = "bulbul:v2"
    target_language_code: str = Field(default="hi-IN", description="BCP-47 language code")
    pace: float = Field(default=1.0, ge=0.3, le=3.0, description="Speech pace (0.3-3.0)")
    pitch: float = Field(default=0.0, ge=-0.75, le=0.75, description="Pitch (only for v2)")
    loudness: float = Field(default=1.0, ge=0.3, le=3.0, description="Loudness (only for v2)")
    sample_rate: int = Field(default=22050, description="Audio sample rate")
    enable_preprocessing: bool = Field(default=True, description="Normalize text")
    temperature: float = Field(default=0.6, ge=0.01, le=1.0, description="Temperature (only for v3-beta)")


class SarvamTTSProvider:
    """
    Sarvam AI Text-to-Speech Provider
    
    Usage:
        provider = SarvamTTSProvider(api_key="your-api-key")
        audio_path = await provider.synthesize(
            text="Hello, this is a test",
            voice_id="Anushka",
            output_path="output.wav"
        )
    """
    
    BASE_URL = "https://api.sarvam.ai/text-to-speech"
    
    def __init__(self, config: Optional[SarvamTTSConfig] = None):
        self.config = config or SarvamTTSConfig()
        if not self.config.api_key:
            raise ValueError(
                "SARVAM_API_KEY not found. Please set it in your .env file.\n"
                "Get your API key at: https://dashboard.sarvam.ai"
            )
    
    async def synthesize(
        self,
        text: str,
        voice_id: str = "Anushka",
        output_path: Optional[str] = None,
        model: Optional[str] = None,
        language_code: Optional[str] = None,
        pace: Optional[float] = None,
        pitch: Optional[float] = None,
        loudness: Optional[float] = None,
        sample_rate: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Convert text to speech using Sarvam AI TTS API
        
        Args:
            text: Text to convert to speech (max 1500 chars for v2, 2500 for v3-beta)
            voice_id: Speaker voice ID (e.g., "Anushka", "Abhilash", "Aditya")
            output_path: Path to save the audio file (.wav)
            model: TTS model to use (bulbul:v2 or bulbul:v3-beta)
            language_code: BCP-47 language code (e.g., "hi-IN", "en-IN")
            pace: Speech pace (0.3-3.0 for v2, 0.5-2.0 for v3-beta)
            pitch: Pitch adjustment (only for v2, -0.75 to 0.75)
            loudness: Loudness adjustment (only for v2, 0.3 to 3.0)
            sample_rate: Audio sample rate (8000, 16000, 22050, 24000)
            temperature: Randomness control (only for v3-beta, 0.01 to 1.0)
            
        Returns:
            Path to the generated audio file
        """
        # Use config defaults if not specified
        model = model or self.config.model
        language_code = language_code or self.config.target_language_code
        pace = pace if pace is not None else self.config.pace
        sample_rate = sample_rate or self.config.sample_rate
        
        # Sarvam API requires lowercase speaker names
        voice_id_normalized = voice_id.lower()
        
        # Build request payload
        payload = {
            "inputs": [text],
            "target_language_code": language_code,
            "speaker": voice_id_normalized,
            "model": model,
            "pace": pace,
            "sample_rate": sample_rate,
            "enable_preprocessing": self.config.enable_preprocessing,
        }
        
        # Add model-specific parameters
        if model == "bulbul:v2":
            payload["pitch"] = pitch if pitch is not None else self.config.pitch
            payload["loudness"] = loudness if loudness is not None else self.config.loudness
        elif model == "bulbul:v3-beta":
            payload["temperature"] = temperature if temperature is not None else self.config.temperature
            # v3-beta has stricter pace limits
            if payload["pace"] < 0.5:
                payload["pace"] = 0.5
            elif payload["pace"] > 2.0:
                payload["pace"] = 2.0
        
        headers = {
            "Content-Type": "application/json",
            "api-subscription-key": self.config.api_key,
        }
        
        logger.debug(f"Sarvam TTS request: voice={voice_id}, model={model}, text_length={len(text)}")
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                self.BASE_URL,
                json=payload,
                headers=headers,
            )
            
            if response.status_code != 200:
                error_detail = response.text
                logger.error(f"Sarvam TTS API error: {response.status_code} - {error_detail}")
                raise Exception(f"Sarvam TTS API error: {response.status_code} - {error_detail}")
            
            result = response.json()
        
        # Decode base64 audio
        if "audios" not in result or not result["audios"]:
            raise Exception("No audio returned from Sarvam TTS API")
        
        audio_base64 = result["audios"][0]
        audio_bytes = base64.b64decode(audio_base64)
        
        # Generate output path if not provided
        if not output_path:
            output_path = f"sarvam_tts_output_{voice_id}.wav"
        
        # Ensure directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Write audio file
        with open(output_path, "wb") as f:
            f.write(audio_bytes)
        
        logger.info(f"Sarvam TTS: Generated audio saved to {output_path}")
        return output_path
    
    async def synthesize_batch(
        self,
        segments: list[dict],
        output_dir: str,
        model: Optional[str] = None,
        language_code: Optional[str] = None,
    ) -> list[str]:
        """
        Convert multiple text segments to speech
        
        Args:
            segments: List of dicts with 'text' and 'voice_id' keys
            output_dir: Directory to save audio files
            model: TTS model to use
            language_code: BCP-47 language code
            
        Returns:
            List of paths to generated audio files
        """
        output_paths = []
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        for i, segment in enumerate(segments):
            text = segment.get("text", "")
            voice_id = segment.get("voice_id", "Anushka")
            
            if not text.strip():
                continue
            
            output_path = str(Path(output_dir) / f"segment_{i:04d}_{voice_id}.wav")
            
            await self.synthesize(
                text=text,
                voice_id=voice_id,
                output_path=output_path,
                model=model,
                language_code=language_code,
            )
            
            output_paths.append(output_path)
        
        return output_paths


# Available voices for reference
SARVAM_VOICES_V2 = {
    "female": ["Anushka", "Manisha", "Vidya", "Arya"],
    "male": ["Abhilash", "Karun", "Hitesh"],
}

SARVAM_VOICES_V3 = [
    "Aditya", "Ritu", "Priya", "Neha", "Rahul", "Pooja", "Rohan", "Simran",
    "Kavya", "Amit", "Dev", "Ishita", "Shreya", "Ratan", "Varun", "Manan",
    "Sumit", "Roopa", "Kabir", "Aayan", "Shubh", "Ashutosh", "Advait",
    "Amelia", "Sophia"
]

# Convenience function
async def create_sarvam_tts(api_key: Optional[str] = None) -> SarvamTTSProvider:
    """Create a Sarvam TTS provider instance"""
    config = SarvamTTSConfig()
    if api_key:
        config.api_key = api_key
    return SarvamTTSProvider(config)
