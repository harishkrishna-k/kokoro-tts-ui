import io
import sys
import wave
from pathlib import Path
from types import ModuleType

import streamlit as st
import numpy as np

# transformers 5.x has hard imports of torchvision submodules in its
# AlbertModel import chain, even though kokoro doesn't need vision.
# Mock the entire torchvision namespace to prevent ModuleNotFoundError.
from importlib.util import spec_from_loader
from unittest.mock import MagicMock

class _MockModule(ModuleType):
    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        return MagicMock()

def _make_tv(name):
    m = _MockModule(name)
    m.__path__ = []
    m.__package__ = name.rpartition('.')[0] if '.' in name else name
    spec = spec_from_loader(name, loader=None)
    spec.submodule_search_locations = []
    m.__spec__ = spec
    return m

if 'torchvision' not in sys.modules:
    sys.modules['torchvision'] = _make_tv('torchvision')
    for _sub in ('io', 'transforms', 'transforms.functional'):
        sys.modules[f'torchvision.{_sub}'] = _make_tv(f'torchvision.{_sub}')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

LANGUAGES = {
    "a": "American English",
    "b": "British English",
    "e": "Spanish",
    "f": "French",
    "h": "Hindi",
    "i": "Italian",
    "p": "Brazilian Portuguese",
    "j": "Japanese",
    "z": "Mandarin Chinese",
}

VOICES = {
    "a": {
        "Female": ["af_heart", "af_bella", "af_nicole", "af_aoede", "af_kore", "af_sarah", "af_nova", "af_sky", "af_alloy", "af_jessica", "af_river"],
        "Male": ["am_michael", "am_fenrir", "am_puck", "am_echo", "am_eric", "am_liam", "am_onyx", "am_santa", "am_adam"],
    },
    "b": {
        "Female": ["bf_emma", "bf_isabella", "bf_alice", "bf_lily"],
        "Male": ["bm_george", "bm_fable", "bm_lewis", "bm_daniel"],
    },
    "e": {
        "Female": ["ef_dora"],
        "Male": ["em_alex", "em_santa"],
    },
    "f": {
        "Female": ["ff_siwis"],
    },
    "h": {
        "Female": ["hf_alpha", "hf_beta"],
    },
    "i": {
        "Female": ["if_sara"],
        "Male": ["im_nicola"],
    },
    "p": {
        "Female": ["pf_dora"],
        "Male": ["pm_alex", "pm_santa"],
    },
    "j": {
        "Female": ["jf_alpha", "jf_tebukuro", "jf_nezumi", "jf_gongitsune"],
        "Male": ["jm_kumo"],
    },
    "z": {
        "Female": ["zf_xiaoni", "zf_xiaoyi", "zf_xiaobei", "zf_xiaoxiao"],
        "Male": ["zm_yunxi", "zm_yunyang", "zm_yunjian", "zm_yunxia"],
    },
}


def get_all_voices(lang_code: str) -> list[str]:
    voices = []
    for group in VOICES.get(lang_code, {}).values():
        voices.extend(group)
    return voices


def generate_audio(text: str, lang_code: str, voice: str, speed: float):
    from kokoro import KPipeline

    pipeline = KPipeline(lang_code=lang_code)
    for result in pipeline(text, voice=voice, speed=speed, split_pattern=r"\n+"):
        if result.audio is not None:
            yield result.audio


def audio_to_wav_bytes(audio_tensors: list) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)
        for audio in audio_tensors:
            audio_np = audio.numpy()
            audio_int16 = (audio_np * 32767).astype(np.int16)
            wf.writeframes(audio_int16.tobytes())
    return buf.getvalue()


st.set_page_config(page_title="Kokoro TTS", page_icon="🔊", layout="centered")

st.title("🔊 Kokoro TTS")
st.markdown("Generate speech from text using [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)")

with st.sidebar:
    st.header("Settings")
    lang_code = st.selectbox(
        "Language",
        options=list(LANGUAGES.keys()),
        format_func=lambda x: LANGUAGES[x],
        index=0,
    )

    voices = get_all_voices(lang_code)
    voice = st.selectbox("Voice", voices, index=0)

    speed = st.slider("Speed", min_value=0.5, max_value=2.0, value=1.0, step=0.1)

st.divider()

input_method = st.radio("Input method", ["Paste text", "Upload file"], horizontal=True)

text = ""
if input_method == "Paste text":
    text = st.text_area("Enter text to synthesize", height=200, placeholder="Type or paste your text here...")
else:
    uploaded = st.file_uploader("Upload a .txt or .md file", type=["txt", "md"])
    if uploaded:
        text = uploaded.read().decode("utf-8")
        with st.expander("Preview"):
            st.text(text[:2000] + ("..." if len(text) > 2000 else ""))

if st.button("Generate Audio", type="primary", use_container_width=True):
    if not text.strip():
        st.warning("Please enter or upload some text.")
        st.stop()

    with st.spinner("Generating audio..."):
        try:
            audio_chunks = list(generate_audio(text.strip(), lang_code, voice, speed))
        except Exception as e:
            st.error(f"Generation failed: {e}")
            st.stop()

    if not audio_chunks:
        st.warning("No audio was generated.")
        st.stop()

    wav_bytes = audio_to_wav_bytes(audio_chunks)
    duration = sum(len(a) for a in audio_chunks) / 24000

    st.success(f"Generated {duration:.1f}s of audio")
    st.audio(wav_bytes, format="audio/wav")
    st.download_button(
        label="Download WAV",
        data=wav_bytes,
        file_name="kokoro_output.wav",
        mime="audio/wav",
        use_container_width=True,
    )
