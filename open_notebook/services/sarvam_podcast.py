"""
Sarvam Podcast Generator

A custom podcast generator that uses:
- Groq for LLM (script/transcript generation)
- Sarvam AI for TTS (text-to-speech)

This bypasses the default TTS providers and uses Sarvam directly.
"""

import asyncio
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from loguru import logger
from pydantic import BaseModel

from open_notebook.config import DATA_FOLDER
from open_notebook.services.sarvam_tts import SarvamTTSProvider, SarvamTTSConfig


class SpeakerConfig(BaseModel):
    """Configuration for a podcast speaker"""
    name: str
    voice_id: str  # Sarvam voice ID (e.g., "Anushka", "Abhilash", "Aditya")
    backstory: str
    personality: str


class TranscriptSegment(BaseModel):
    """A single segment of the podcast transcript"""
    speaker: str
    text: str
    voice_id: Optional[str] = None


class PodcastTranscript(BaseModel):
    """Complete podcast transcript"""
    title: str
    segments: List[TranscriptSegment]


class SarvamPodcastGenerator:
    """
    Podcast generator using Groq for LLM and Sarvam for TTS
    
    Usage:
        generator = SarvamPodcastGenerator(
            groq_api_key="gsk_...",
            sarvam_api_key="..."
        )
        result = await generator.generate_podcast(
            content="Your content here",
            episode_name="My Episode",
            speakers=[speaker1, speaker2],
            output_dir="./output"
        )
    """
    
    def __init__(
        self,
        groq_api_key: Optional[str] = None,
        sarvam_api_key: Optional[str] = None,
        groq_model: str = "llama-3.3-70b-versatile",
        sarvam_model: str = "bulbul:v2",
        language_code: str = "en-IN",
    ):
        self.groq_api_key = groq_api_key or os.environ.get("GROQ_API_KEY", "")
        self.sarvam_api_key = sarvam_api_key or os.environ.get("SARVAM_API_KEY", "")
        self.groq_model = groq_model
        self.sarvam_model = sarvam_model
        self.language_code = language_code
        
        if not self.groq_api_key:
            raise ValueError(
                "GROQ_API_KEY not found. Please set it in your .env file.\n"
                "Get your key at: https://console.groq.com/keys"
            )
        
        if not self.sarvam_api_key:
            raise ValueError(
                "SARVAM_API_KEY not found. Please set it in your .env file.\n"
                "Get your key at: https://dashboard.sarvam.ai"
            )
        
        # Initialize Sarvam TTS
        sarvam_config = SarvamTTSConfig(
            api_key=self.sarvam_api_key,
            model=self.sarvam_model,
            target_language_code=self.language_code,
        )
        self.tts_provider = SarvamTTSProvider(sarvam_config)
    
    async def generate_transcript(
        self,
        content: str,
        speakers: List[SpeakerConfig],
        briefing: str = "",
        num_segments: int = 10,
    ) -> PodcastTranscript:
        """
        Generate podcast transcript from content using Groq LLM
        
        Args:
            content: Source content to convert to podcast
            speakers: List of speaker configurations
            briefing: Additional instructions for script generation
            num_segments: Approximate number of dialogue segments
            
        Returns:
            PodcastTranscript with segments
        """
        speaker_descriptions = "\n".join([
            f"- {s.name}: {s.personality}. Background: {s.backstory}"
            for s in speakers
        ])
        
        speaker_names = [s.name for s in speakers]
        
        prompt = f"""You are a podcast script writer. Create an engaging podcast conversation between the following speakers:

{speaker_descriptions}

Based on the following content, create a natural, conversational podcast script. The conversation should:
- Be informative and engaging
- Have natural back-and-forth dialogue
- Include the speakers' personalities
- Cover the key points from the content
- Have approximately {num_segments} dialogue exchanges

{f"Additional instructions: {briefing}" if briefing else ""}

CONTENT TO DISCUSS:
{content}

OUTPUT FORMAT:
Output ONLY a JSON object with this exact structure (no markdown, no code blocks):
{{
    "title": "Episode Title",
    "segments": [
        {{"speaker": "SpeakerName", "text": "What they say..."}},
        {{"speaker": "OtherSpeaker", "text": "Their response..."}}
    ]
}}

Use ONLY these speaker names: {', '.join(speaker_names)}

Begin generating the podcast script:"""

        logger.info(f"Generating transcript with Groq model: {self.groq_model}")
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.groq_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.groq_model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a podcast script writer. Output valid JSON only."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 4000,
                },
            )
            
            if response.status_code != 200:
                raise Exception(f"Groq API error: {response.status_code} - {response.text}")
            
            result = response.json()
        
        # Extract the content
        content_text = result["choices"][0]["message"]["content"]
        
        # Clean up the response - remove markdown code blocks if present
        content_text = re.sub(r'^```json\s*', '', content_text.strip())
        content_text = re.sub(r'^```\s*', '', content_text.strip())
        content_text = re.sub(r'\s*```$', '', content_text.strip())
        
        # Parse JSON
        import json
        try:
            transcript_data = json.loads(content_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse transcript JSON: {e}")
            logger.error(f"Raw content: {content_text[:500]}...")
            raise Exception(f"Failed to parse transcript: {e}")
        
        # Map speaker names to voice IDs
        speaker_voice_map = {s.name: s.voice_id for s in speakers}
        
        segments = []
        for seg in transcript_data.get("segments", []):
            speaker_name = seg.get("speaker", speakers[0].name)
            voice_id = speaker_voice_map.get(speaker_name, speakers[0].voice_id)
            segments.append(TranscriptSegment(
                speaker=speaker_name,
                text=seg.get("text", ""),
                voice_id=voice_id,
            ))
        
        logger.info(f"Generated transcript with {len(segments)} segments")
        
        return PodcastTranscript(
            title=transcript_data.get("title", "Untitled Episode"),
            segments=segments,
        )
    
    async def generate_audio(
        self,
        transcript: PodcastTranscript,
        output_dir: str,
        batch_size: int = 3,
    ) -> List[str]:
        """
        Convert transcript segments to audio using Sarvam TTS
        
        Args:
            transcript: Podcast transcript with segments
            output_dir: Directory to save audio files
            batch_size: Number of concurrent TTS requests
            
        Returns:
            List of paths to generated audio files
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        audio_files = []
        
        logger.info(f"Generating audio for {len(transcript.segments)} segments")
        
        # Process in batches to avoid rate limiting
        for i in range(0, len(transcript.segments), batch_size):
            batch = transcript.segments[i:i + batch_size]
            
            tasks = []
            for j, segment in enumerate(batch):
                idx = i + j
                output_path = str(Path(output_dir) / f"segment_{idx:04d}.wav")
                
                if not segment.text.strip():
                    continue
                
                task = self.tts_provider.synthesize(
                    text=segment.text,
                    voice_id=segment.voice_id or "Anushka",
                    output_path=output_path,
                )
                tasks.append((output_path, task))
            
            # Execute batch
            for output_path, task in tasks:
                try:
                    await task
                    audio_files.append(output_path)
                except Exception as e:
                    logger.error(f"Failed to generate audio segment: {e}")
            
            # Small delay between batches to avoid rate limiting
            if i + batch_size < len(transcript.segments):
                await asyncio.sleep(0.5)
        
        logger.info(f"Generated {len(audio_files)} audio segments")
        return audio_files
    
    async def merge_audio_files(
        self,
        audio_files: List[str],
        output_path: str,
        gap_ms: int = 500,
    ) -> str:
        """
        Merge multiple audio files into a single podcast file
        
        Args:
            audio_files: List of audio file paths
            output_path: Path for the merged output file
            gap_ms: Gap between segments in milliseconds
            
        Returns:
            Path to the merged audio file
        """
        if not audio_files:
            raise ValueError("No audio files to merge")
        
        if len(audio_files) == 1:
            # Just copy the single file
            import shutil
            shutil.copy(audio_files[0], output_path)
            return output_path
        
        # Try using ffmpeg if available
        try:
            # Create a file list for ffmpeg
            list_file = Path(output_path).parent / "concat_list.txt"
            with open(list_file, "w") as f:
                for audio_file in audio_files:
                    # Escape single quotes for ffmpeg
                    escaped_path = audio_file.replace("'", "'\"'\"'")
                    f.write(f"file '{escaped_path}'\n")
            
            # Run ffmpeg to concatenate
            cmd = [
                "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                "-i", str(list_file), "-c", "copy", output_path
            ]
            
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300
            )
            
            if result.returncode != 0:
                logger.warning(f"ffmpeg concatenation failed: {result.stderr}")
                raise Exception("ffmpeg failed")
            
            # Clean up
            list_file.unlink()
            
            logger.info(f"Merged audio saved to: {output_path}")
            return output_path
            
        except (FileNotFoundError, Exception) as e:
            logger.warning(f"ffmpeg not available or failed: {e}")
            logger.info("Falling back to simple concatenation")
            
            # Simple concatenation for WAV files
            with open(output_path, "wb") as output:
                for i, audio_file in enumerate(audio_files):
                    with open(audio_file, "rb") as f:
                        if i == 0:
                            # Include header from first file
                            output.write(f.read())
                        else:
                            # Skip WAV header (44 bytes) for subsequent files
                            f.seek(44)
                            output.write(f.read())
            
            return output_path
    
    async def generate_podcast(
        self,
        content: str,
        episode_name: str,
        speakers: List[SpeakerConfig],
        output_dir: Optional[str] = None,
        briefing: str = "",
        num_segments: int = 10,
    ) -> Dict[str, Any]:
        """
        Generate a complete podcast from content
        
        Args:
            content: Source content to convert to podcast
            episode_name: Name of the episode
            speakers: List of speaker configurations
            output_dir: Directory to save output files
            briefing: Additional instructions for script generation
            num_segments: Approximate number of dialogue segments
            
        Returns:
            Dictionary with transcript, audio_files, and final_output_path
        """
        # Set default output directory
        if not output_dir:
            output_dir = str(Path(DATA_FOLDER) / "podcasts" / "episodes" / episode_name)
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting podcast generation: {episode_name}")
        logger.info(f"Using {len(speakers)} speakers")
        logger.info(f"Output directory: {output_dir}")
        
        # Step 1: Generate transcript
        logger.info("Step 1/3: Generating transcript with Groq...")
        transcript = await self.generate_transcript(
            content=content,
            speakers=speakers,
            briefing=briefing,
            num_segments=num_segments,
        )
        
        # Save transcript
        import json
        transcript_path = str(Path(output_dir) / "transcript.json")
        with open(transcript_path, "w") as f:
            json.dump(transcript.model_dump(), f, indent=2)
        logger.info(f"Saved transcript to: {transcript_path}")
        
        # Step 2: Generate audio segments
        logger.info("Step 2/3: Generating audio with Sarvam TTS...")
        segments_dir = str(Path(output_dir) / "segments")
        audio_files = await self.generate_audio(
            transcript=transcript,
            output_dir=segments_dir,
        )
        
        # Step 3: Merge audio files
        logger.info("Step 3/3: Merging audio files...")
        final_audio_path = str(Path(output_dir) / f"{episode_name}.wav")
        await self.merge_audio_files(
            audio_files=audio_files,
            output_path=final_audio_path,
        )
        
        logger.info(f"Podcast generation complete: {final_audio_path}")
        
        return {
            "transcript": transcript.model_dump(),
            "transcript_path": transcript_path,
            "audio_files": audio_files,
            "final_output_path": final_audio_path,
        }


async def generate_podcast_with_sarvam(
    content: str,
    episode_name: str,
    speaker1_name: str = "Host",
    speaker1_voice: str = "Anushka",
    speaker2_name: str = "Guest", 
    speaker2_voice: str = "Abhilash",
    briefing: str = "",
    output_dir: Optional[str] = None,
    groq_model: str = "llama-3.3-70b-versatile",
    sarvam_model: str = "bulbul:v2",
    language_code: str = "en-IN",
) -> Dict[str, Any]:
    """
    Convenience function to generate a podcast with default 2 speakers
    
    Args:
        content: Source content
        episode_name: Episode name
        speaker1_name: First speaker's name
        speaker1_voice: Sarvam voice ID for first speaker
        speaker2_name: Second speaker's name
        speaker2_voice: Sarvam voice ID for second speaker
        briefing: Additional instructions
        output_dir: Output directory
        groq_model: Groq model to use
        sarvam_model: Sarvam TTS model
        language_code: BCP-47 language code
        
    Returns:
        Generation result dictionary
    """
    speakers = [
        SpeakerConfig(
            name=speaker1_name,
            voice_id=speaker1_voice,
            backstory="The main host of the podcast",
            personality="Friendly, curious, and engaging",
        ),
        SpeakerConfig(
            name=speaker2_name,
            voice_id=speaker2_voice,
            backstory="The expert guest on the topic",
            personality="Knowledgeable, clear, and informative",
        ),
    ]
    
    generator = SarvamPodcastGenerator(
        groq_model=groq_model,
        sarvam_model=sarvam_model,
        language_code=language_code,
    )
    
    return await generator.generate_podcast(
        content=content,
        episode_name=episode_name,
        speakers=speakers,
        output_dir=output_dir,
        briefing=briefing,
    )
