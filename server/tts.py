import asyncio
import edge_tts

VOICES = {
    "male": "en-US-GuyNeural",
    "female": "en-US-JennyNeural",
}


async def _synth(text: str, voice: str, out_path: str):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(out_path)


def synthesize(text: str, gender: str, out_path: str) -> str:
    gender = (gender or "female").lower()
    if gender not in VOICES:
        raise ValueError(f"gender must be one of {list(VOICES)}")
    voice = VOICES[gender]
    asyncio.run(_synth(text, voice, out_path))
    return out_path
