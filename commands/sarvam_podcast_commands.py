"""
Sarvam Podcast Commands

Custom podcast generation command that uses:
- Groq for LLM (script generation)
- Sarvam AI for TTS (text-to-speech)
"""

import time
from pathlib import Path
from typing import Optional

from loguru import logger
from pydantic import BaseModel
from surreal_commands import CommandInput, CommandOutput, command

from open_notebook.config import DATA_FOLDER
from open_notebook.database.repository import ensure_record_id, repo_query
from open_notebook.domain.podcast import EpisodeProfile, PodcastEpisode, SpeakerProfile


def full_model_dump(model):
    if isinstance(model, BaseModel):
        return model.model_dump()
    elif isinstance(model, dict):
        return {k: full_model_dump(v) for k, v in model.items()}
    elif isinstance(model, list):
        return [full_model_dump(item) for item in model]
    else:
        return model


class SarvamPodcastInput(CommandInput):
    """Input for Sarvam podcast generation"""
    episode_profile: str
    speaker_profile: str
    episode_name: str
    content: str
    briefing_suffix: Optional[str] = None
    groq_model: str = "llama-3.3-70b-versatile"
    sarvam_model: str = "bulbul:v2"
    language_code: str = "en-IN"


class SarvamPodcastOutput(CommandOutput):
    """Output for Sarvam podcast generation"""
    success: bool
    episode_id: Optional[str] = None
    audio_file_path: Optional[str] = None
    transcript: Optional[dict] = None
    processing_time: float
    error_message: Optional[str] = None


@command("generate_podcast_sarvam", app="open_notebook")
async def generate_podcast_sarvam_command(
    input_data: SarvamPodcastInput,
) -> SarvamPodcastOutput:
    """
    Generate podcast using Groq for LLM and Sarvam for TTS
    """
    start_time = time.time()
    
    try:
        from open_notebook.services.sarvam_podcast import (
            SarvamPodcastGenerator,
            SpeakerConfig,
        )
        
        logger.info(
            f"Starting Sarvam podcast generation for episode: {input_data.episode_name}"
        )
        
        # 1. Load Episode and Speaker profiles from SurrealDB
        episode_profile = await EpisodeProfile.get_by_name(input_data.episode_profile)
        if not episode_profile:
            raise ValueError(
                f"Episode profile '{input_data.episode_profile}' not found"
            )
        
        speaker_profile = await SpeakerProfile.get_by_name(
            episode_profile.speaker_config
        )
        if not speaker_profile:
            raise ValueError(
                f"Speaker profile '{episode_profile.speaker_config}' not found"
            )
        
        logger.info(f"Loaded episode profile: {episode_profile.name}")
        logger.info(f"Loaded speaker profile: {speaker_profile.name}")
        
        # 2. Convert speaker profile to SpeakerConfig list
        speakers = []
        for speaker_data in speaker_profile.speakers:
            speakers.append(SpeakerConfig(
                name=speaker_data.get("name", "Speaker"),
                voice_id=speaker_data.get("voice_id", "Anushka"),
                backstory=speaker_data.get("backstory", ""),
                personality=speaker_data.get("personality", ""),
            ))
        
        # 3. Generate briefing
        briefing = episode_profile.default_briefing
        if input_data.briefing_suffix:
            briefing += f"\n\nAdditional instructions: {input_data.briefing_suffix}"
        
        # 4. Create episode record
        episode = PodcastEpisode(
            name=input_data.episode_name,
            episode_profile=full_model_dump(episode_profile.model_dump()),
            speaker_profile=full_model_dump(speaker_profile.model_dump()),
            command=ensure_record_id(input_data.execution_context.command_id)
            if input_data.execution_context
            else None,
            briefing=briefing,
            content=input_data.content,
            audio_file=None,
            transcript=None,
            outline=None,
        )
        await episode.save()
        
        # 5. Create output directory
        output_dir = Path(f"{DATA_FOLDER}/podcasts/episodes/{input_data.episode_name}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 6. Generate podcast using Sarvam
        logger.info("Starting podcast generation with Groq + Sarvam TTS...")
        
        generator = SarvamPodcastGenerator(
            groq_model=input_data.groq_model,
            sarvam_model=input_data.sarvam_model,
            language_code=input_data.language_code,
        )
        
        result = await generator.generate_podcast(
            content=input_data.content,
            episode_name=input_data.episode_name,
            speakers=speakers,
            output_dir=str(output_dir),
            briefing=briefing,
            num_segments=episode_profile.num_segments,
        )
        
        # 7. Update episode record
        episode.audio_file = result.get("final_output_path")
        episode.transcript = {"transcript": result.get("transcript")}
        await episode.save()
        
        processing_time = time.time() - start_time
        logger.info(
            f"Successfully generated Sarvam podcast episode: {episode.id} in {processing_time:.2f}s"
        )
        
        return SarvamPodcastOutput(
            success=True,
            episode_id=str(episode.id),
            audio_file_path=result.get("final_output_path"),
            transcript={"transcript": result.get("transcript")},
            processing_time=processing_time,
        )
    
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Sarvam podcast generation failed: {e}")
        logger.exception(e)
        
        return SarvamPodcastOutput(
            success=False,
            processing_time=processing_time,
            error_message=str(e),
        )
