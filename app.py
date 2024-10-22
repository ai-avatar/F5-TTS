import re
import torch
import torchaudio
import numpy as np
import tempfile
from einops import rearrange
from vocos import Vocos
from pydub import AudioSegment, silence
from model import CFM, DiT
from cached_path import cached_path
from model.utils import (
    load_checkpoint,
    get_tokenizer,
    convert_char_to_pinyin,
    save_spectrogram,
)
import soundfile as sf
import runpod
import tqdm
import boto3
import io
import os
import requests

S3_ENDPOINT = os.getenv('S3_ENDPOINT')
S3_ACCESS_KEY = os.getenv('S3_ACCESS_KEY')
S3_SECRET_KEY = os.getenv('S3_SECRET_KEY')

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

print(f"Using {device} device")

vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz")

# --------------------- Settings -------------------- #

target_sample_rate = 24000
n_mel_channels = 100
hop_length = 256
target_rms = 0.1
nfe_step = 32  # 16, 32
cfg_strength = 2.0
ode_method = "euler"
sway_sampling_coef = -1.0
speed = 1.0
fix_duration = None


def load_model(repo_name, exp_name, model_cls, model_cfg, ckpt_step):
    ckpt_path = str(cached_path(f"hf://SWivid/{repo_name}/{exp_name}/model_{ckpt_step}.safetensors"))
    # ckpt_path = f"ckpts/{exp_name}/model_{ckpt_step}.pt"  # .pt | .safetensors
    vocab_char_map, vocab_size = get_tokenizer("Emilia_ZH_EN", "pinyin")
    model = CFM(
        transformer=model_cls(
            **model_cfg, text_num_embeds=vocab_size, mel_dim=n_mel_channels
        ),
        mel_spec_kwargs=dict(
            target_sample_rate=target_sample_rate,
            n_mel_channels=n_mel_channels,
            hop_length=hop_length,
        ),
        odeint_kwargs=dict(
            method=ode_method,
        ),
        vocab_char_map=vocab_char_map,
    ).to(device)

    model = load_checkpoint(model, ckpt_path, device, use_ema = True)

    return model


# load models
F5TTS_model_cfg = dict(
    dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4
)

F5TTS_ema_model = load_model(
    "F5-TTS", "F5TTS_Base", DiT, F5TTS_model_cfg, 1200000
)

def chunk_text(text, max_chars=135):
    """
    Splits the input text into chunks, each with a maximum number of characters.

    Args:
        text (str): The text to be split.
        max_chars (int): The maximum number of characters per chunk.

    Returns:
        List[str]: A list of text chunks.
    """
    chunks = []
    current_chunk = ""
    # Split the text into sentences based on punctuation followed by whitespace
    sentences = re.split(r'(?<=[;:,.!?])\s+|(?<=[；：，。！？])', text)

    for sentence in sentences:
        if len(current_chunk.encode('utf-8')) + len(sentence.encode('utf-8')) <= max_chars:
            current_chunk += sentence + " " if sentence and len(sentence[-1].encode('utf-8')) == 1 else sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " " if sentence and len(sentence[-1].encode('utf-8')) == 1 else sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def infer_batch(ref_audio, ref_text, gen_text_batches, model, remove_silence, cross_fade_duration=0.15):
    if model == "F5-TTS":
        ema_model = load_model(model, "F5TTS_Base", DiT, F5TTS_model_cfg, 1200000)

    audio, sr = ref_audio
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)

    rms = torch.sqrt(torch.mean(torch.square(audio)))
    if rms < target_rms:
        audio = audio * target_rms / rms
    if sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
        audio = resampler(audio)
    audio = audio.to(device)

    generated_waves = []
    spectrograms = []

    for i, gen_text in enumerate(tqdm.tqdm(gen_text_batches)):
        # Prepare the text
        if len(ref_text[-1].encode('utf-8')) == 1:
            ref_text = ref_text + " "
        text_list = [ref_text + gen_text]
        final_text_list = convert_char_to_pinyin(text_list)

        # Calculate duration
        ref_audio_len = audio.shape[-1] // hop_length
        zh_pause_punc = r"。，、；：？！"
        ref_text_len = len(ref_text.encode('utf-8')) + 3 * len(re.findall(zh_pause_punc, ref_text))
        gen_text_len = len(gen_text.encode('utf-8')) + 3 * len(re.findall(zh_pause_punc, gen_text))
        duration = ref_audio_len + int(ref_audio_len / ref_text_len * gen_text_len / speed)

        # inference
        with torch.inference_mode():
            generated, _ = ema_model.sample(
                cond=audio,
                text=final_text_list,
                duration=duration,
                steps=nfe_step,
                cfg_strength=cfg_strength,
                sway_sampling_coef=sway_sampling_coef,
            )

        generated = generated.to(torch.float32)
        generated = generated[:, ref_audio_len:, :]
        generated_mel_spec = rearrange(generated, "1 n d -> 1 d n")
        generated_wave = vocos.decode(generated_mel_spec.cpu())
        if rms < target_rms:
            generated_wave = generated_wave * rms / target_rms

        # wav -> numpy
        generated_wave = generated_wave.squeeze().cpu().numpy()
        
        generated_waves.append(generated_wave)
        spectrograms.append(generated_mel_spec[0].cpu().numpy())

    # Combine all generated waves with cross-fading
    if cross_fade_duration <= 0:
        # Simply concatenate
        final_wave = np.concatenate(generated_waves)
    else:
        final_wave = generated_waves[0]
        for i in range(1, len(generated_waves)):
            prev_wave = final_wave
            next_wave = generated_waves[i]

            # Calculate cross-fade samples, ensuring it does not exceed wave lengths
            cross_fade_samples = int(cross_fade_duration * target_sample_rate)
            cross_fade_samples = min(cross_fade_samples, len(prev_wave), len(next_wave))

            if cross_fade_samples <= 0:
                # No overlap possible, concatenate
                final_wave = np.concatenate([prev_wave, next_wave])
                continue

            # Overlapping parts
            prev_overlap = prev_wave[-cross_fade_samples:]
            next_overlap = next_wave[:cross_fade_samples]

            # Fade out and fade in
            fade_out = np.linspace(1, 0, cross_fade_samples)
            fade_in = np.linspace(0, 1, cross_fade_samples)

            # Cross-faded overlap
            cross_faded_overlap = prev_overlap * fade_out + next_overlap * fade_in

            # Combine
            new_wave = np.concatenate([
                prev_wave[:-cross_fade_samples],
                cross_faded_overlap,
                next_wave[cross_fade_samples:]
            ])

            final_wave = new_wave

    # Create a combined spectrogram
    combined_spectrogram = np.concatenate(spectrograms, axis=1)

    return final_wave, combined_spectrogram

def infer(ref_audio, ref_text, gen_text, model, remove_silence, cross_fade_duration=0.15):
    print(gen_text)

    # Add the functionality to ensure it ends with ". "
    if not ref_text.endswith(". "):
        if ref_text.endswith("."):
            ref_text += " "
        else:
            ref_text += ". "

    audio, sr = torchaudio.load(ref_audio)

    # Use the new chunk_text function to split gen_text
    max_chars = int(len(ref_text.encode('utf-8')) / (audio.shape[-1] / sr) * (25 - audio.shape[-1] / sr))
    gen_text_batches = chunk_text(gen_text, max_chars=max_chars)
    print('ref_text', ref_text)
    for i, batch_text in enumerate(gen_text_batches):
        print(f'gen_text {i}', batch_text)
    
    print(f"Generating audio using {model} in {len(gen_text_batches)} batches")
    return infer_batch((audio, sr), ref_text, gen_text_batches, model, remove_silence, cross_fade_duration)

def process_voice(aseg, ref_text):
    print("Converting audio...")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        non_silent_segs = silence.split_on_silence(aseg, min_silence_len=1000, silence_thresh=-50, keep_silence=1000)
        non_silent_wave = AudioSegment.silent(duration=0)
        for non_silent_seg in non_silent_segs:
            non_silent_wave += non_silent_seg
        aseg = non_silent_wave

        audio_duration = len(aseg)
        if audio_duration > 15000:
            print("Audio is over 15s, clipping to only first 15s.")
            aseg = aseg[:15000]
        aseg.export(f.name, format="wav")
        ref_audio = f.name

    return ref_audio, ref_text

ref_audio1, ref_text1 = process_voice(AudioSegment.from_file("ref_audio/w01.wav"), "Life's a journey, so embrace every moment with an open heart.")
ref_audio2, ref_text2 = process_voice(AudioSegment.from_file("ref_audio/m01.wav"), "Life's a journey, so embrace every moment with an open heart.")

def handler(job):
    job_input = job["input"]
    file_name = job_input["file_name"]
    text = job_input["text"]
    speaker = job_input["speaker"]
    ref_audio = job_input["ref_audio"]
    ref_text = job_input["ref_text"]

    custom_ref_audio = False
    if ref_audio and ref_text:
        res = requests.get(ref_audio)
        ref_audio = AudioSegment.from_file(io.BytesIO(res.content))
        ref_audio, ref_text = process_voice(ref_audio, ref_text)
        custom_ref_audio = True
    elif speaker == "w01":
        ref_audio, ref_text = ref_audio1, ref_text1
    else:
        ref_audio, ref_text = ref_audio2, ref_text2

    remove_silence = False
    audio, _ = infer(ref_audio, ref_text, text, "F5-TTS", remove_silence)

    buffer = io.BytesIO()
    sf.write(buffer, audio, samplerate=target_sample_rate, format='wav')
    buffer.seek(0)

    s3 = boto3.client('s3',
        endpoint_url = S3_ENDPOINT,
        aws_access_key_id = S3_ACCESS_KEY,
        aws_secret_access_key = S3_SECRET_KEY
    )
    s3.upload_fileobj(buffer, "podcasts", file_name)

    if custom_ref_audio:
        os.remove(ref_audio)

    return "Success"

runpod.serverless.start({"handler": handler})