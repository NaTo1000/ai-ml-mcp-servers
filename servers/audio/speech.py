"""
Speech / Audio MCP Server
Transcription (Whisper), basic TTS info, audio feature helpers.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from servers.common import create_server, get_device, safe_json

logger = logging.getLogger(__name__)
mcp = create_server(
    "audio-speech",
    "Speech tools powered by OpenAI Whisper (local): transcribe audio files, "
    "detect language, and return timestamps. TTS requires external providers.",
)

_whisper = None


def _get_whisper(model_size: str = "base"):
    global _whisper
    if _whisper is None or getattr(_whisper, "_size", None) != model_size:
        import whisper
        device = get_device()
        _whisper = whisper.load_model(model_size, device=device)
        _whisper._size = model_size
        logger.info("Loaded Whisper %s on %s", model_size, device)
    return _whisper


@mcp.tool()
def transcribe(
    audio_path: str,
    model_size: str = "base",
    language: Optional[str] = None,
    task: str = "transcribe",
) -> str:
    """Transcribe an audio file (wav/mp3/m4a/...). task='transcribe' or 'translate' (to English)."""
    if not os.path.exists(audio_path):
        return safe_json({"error": f"File not found: {audio_path}"})
    model = _get_whisper(model_size)
    result = model.transcribe(audio_path, language=language, task=task)
    return safe_json({
        "text": result.get("text"),
        "language": result.get("language"),
        "segments": [
            {"start": s["start"], "end": s["end"], "text": s["text"]}
            for s in result.get("segments", [])[:50]
        ],
    })


@mcp.tool()
def detect_language(audio_path: str, model_size: str = "base") -> str:
    """Detect the spoken language of an audio file."""
    import whisper
    model = _get_whisper(model_size)
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    _, probs = model.detect_language(mel)
    top = sorted(probs.items(), key=lambda x: -x[1])[:5]
    return safe_json({"top_languages": [{ "lang": k, "prob": float(v)} for k, v in top]})


@mcp.tool()
def list_whisper_models() -> str:
    """List available Whisper model sizes and approximate VRAM."""
    return safe_json({
        "models": [
            {"size": "tiny", "params": "39M", "vram": "~1 GB"},
            {"size": "base", "params": "74M", "vram": "~1 GB"},
            {"size": "small", "params": "244M", "vram": "~2 GB"},
            {"size": "medium", "params": "769M", "vram": "~5 GB"},
            {"size": "large", "params": "1550M", "vram": "~10 GB"},
            {"size": "large-v3", "params": "1550M", "vram": "~10 GB"},
        ]
    })


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
