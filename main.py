%%capture
!pip install fastapi uvicorn pyngrok torchaudio google-genai edge-tts

%%capture
!apt-get install ffmpeg


import edge_tts
import os
from google import genai
from google.genai import types

import asyncio
import numpy as np
import soundfile as sf
import io
import wave
from scipy import signal
import subprocess

def build_configs():
    edge_compatible_voices = {
    "EN_F": "en-US-JennyNeural",
    "EN_M": "en-US-GuyNeural",
    "AR_F": "ar-SA-ZariyahNeural",
    "AR_M": "ar-SA-HamedNeural",
    "FA_F": "fa-IR-DilaraNeural",
    "FA_M": "fa-IR-FaridNeural"
    }

    configs = {}

    configs["EN_F"] = {
          "system_instruction":"You are a translator. When I send you text, translate it to English and ONLY respond with the English translation. Do not have a conversation, do not ask questions, do not explain - just translate to English using English alphabet only.",
          "target_language": "English",
          "voice_name": edge_compatible_voices["EN_F"]
    }

    configs["EN_M"] = {
          "system_instruction":"You are a translator. When I send you text, translate it to English and ONLY respond with the English translation. Do not have a conversation, do not ask questions, do not explain - just translate to English using English alphabet only.",
          "target_language": "English",
          "voice_name": edge_compatible_voices["EN_M"]
    }

    configs["FA_F"] = {
          "system_instruction":"You are a translator. When I send you text, translate it to Persian and ONLY respond with the Persian translation. Do not have a conversation, do not ask questions, do not explain - just translate to Persian using Persian alphabet only.",
          "target_language": "Persian",
          "voice_name": edge_compatible_voices["FA_F"]
    }

    configs["FA_M"] = {
          "system_instruction":"You are a translator. When I send you text, translate it to Persian and ONLY respond with the Persian translation. Do not have a conversation, do not ask questions, do not explain - just translate to Persian using Persian alphabet only.",
          "target_language": "Persian",
          "voice_name": edge_compatible_voices["FA_M"]
    }

    configs["AR_F"] = {
          "system_instruction":"You are a translator. When I send you text, translate it to Arabic and ONLY respond with the Arabic translation. Do not have a conversation, do not ask questions, do not explain - just translate to Arabic using Arabic alphabet only.",
          "target_language": "Arabic",
          "voice_name": edge_compatible_voices["AR_F"]
    }

    configs["AR_M"] = {
          "system_instruction":"You are a translator. When I send you text, translate it to Arabic and ONLY respond with the Arabic translation. Do not have a conversation, do not ask questions, do not explain - just translate to Arabic using Arabic alphabet only.",
          "target_language": "Arabic",
          "voice_name": edge_compatible_voices["AR_M"]
    }

    return configs


def build_clients():
    # Clients
    clients = {}

    clients["EN_M"] = genai.Client(api_key=os.environ["ENKEY"])
    clients["EN_F"] = genai.Client(api_key=os.environ["ENKEY2"])
    clients["AR_M"] = genai.Client(api_key=os.environ["ARKEY"])
    clients["AR_F"] = genai.Client(api_key=os.environ["ARKEY2"])
    clients["FA_M"] = genai.Client(api_key=os.environ["FAKEY"])
    clients["FA_F"] = genai.Client(api_key=os.environ["FAKEY2"])
    return clients


def generate_text_for_lang(key, sys, text):
    if key not in clients.keys():
        raise ValueError(f"No API key configured for language: {key}")

    # Switch the API key for this call
    client = clients[key]

    response = client.models.generate_content(
    model="gemini-2.0-flash",
    config=types.GenerateContentConfig(
        system_instruction=sys),
    contents=text
    )

    return response.text


async def gemini_translate(key, config, text_input):
    transcript_text = ''
    target_sample_rate = 16000

    try:

        if isinstance(config, dict):
            system_instruction = config.get("system_instruction", "Translate this text.")
            target_language = config.get("target_language", "English")
            voice_name = config.get("voice_name", "en-US-JennyNeural")
        else:
            print("error: wrong config")

        transcript_text = generate_text_for_lang(key, system_instruction, text_input)

        try:
            communicate = edge_tts.Communicate(text=transcript_text, voice=voice_name)

            # Stream audio chunks (MP3 format) into a byte blob
            mp3_chunks = []
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    mp3_chunks.append(chunk["data"])
            mp3_data = b"".join(mp3_chunks)

            # Decode MP3 -> WAV using ffmpeg subprocess (no pydub!)
            ffmpeg = subprocess.Popen(
                ['ffmpeg', '-i', 'pipe:0', '-f', 'wav', 'pipe:1'],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
            )
            wav_data, _ = ffmpeg.communicate(input=mp3_data)

            wav_file = io.BytesIO(wav_data)
            with wave.open(wav_file, 'rb') as wav:
                sample_rate = wav.getframerate()
                n_channels = wav.getnchannels()
                sample_width = wav.getsampwidth()
                raw_audio = wav.readframes(wav.getnframes())

            # Convert to float32 numpy
            if sample_width == 2:
                audio_samples = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32) / 32768.0
            elif sample_width == 4:
                audio_samples = np.frombuffer(raw_audio, dtype=np.int32).astype(np.float32) / 2147483648.0
            else:
                raise ValueError(f"Unsupported sample width: {sample_width}")

            # Stereo → mono if needed
            if n_channels == 2:
                audio_samples = audio_samples.reshape(-1, 2).mean(axis=1)

            # Resample if needed
            if sample_rate != target_sample_rate:
                num_samples = int(len(audio_samples) * target_sample_rate / sample_rate)
                audio_samples = signal.resample(audio_samples, num_samples)
                sample_rate = target_sample_rate

        except Exception as e:
            print(f"[edge_tts_to_float_audio] Error: {e}")
            raise

        return audio_samples, transcript_text, sample_rate

    except Exception as e:
        print(f"Error in gemini_translate: {e}")
        raise


def get_client_key(tgt_lang, speaker_id):
    return {
        "en-US": "EN_F" if speaker_id == 0 else "EN_M",
        "ar-SA": "AR_F" if speaker_id == 0 else "AR_M",
        "fa-IR": "FA_F" if speaker_id == 0 else "FA_M",
    }.get(tgt_lang) or (_ for _ in ()).throw(ValueError(f"Unsupported target language: {tgt_lang}"))


async def t2S_translate(text_input, tgt_lang, speaker_id):
  # Clients anf configs (should differ for each language and speaker)
  key = get_client_key(tgt_lang, speaker_id)

  audio_bytes, translated_text, sample_rate = await gemini_translate(
    key=key,
    config=configs[key],
    text_input=text_input
)
  return audio_bytes, sample_rate, translated_text


# Set up Fast api
import asyncio, time, base64, io
from collections import defaultdict
from fastapi import FastAPI, WebSocket
from typing import Dict, List
from contextlib import asynccontextmanager
from starlette.websockets import WebSocketState
#from file import t2S_translate
current_recorder: WebSocket | None = None

PING_TIMEOUT = 40  # seconds

async def lifespan(app: FastAPI):
    global configs, clients
    configs = build_configs()
    clients = build_clients()

    print("Starting up ...")
    yield
    print("Shutting down. Closing all WebSocket connections...")
    websockets = list(rooms[DEFAULT_ROOM].keys())

    for ws in websockets:
        try:
            await ws.close(code=1001)  # 1001 = Going Away
        except Exception as e:
            print(f"WebSocket Disconnected! {e} ")
        finally:
            rooms[DEFAULT_ROOM].pop(ws, None)

    print("Shutdown complete.")

app = FastAPI(lifespan=lifespan)

DEFAULT_ROOM = "default_room"
rooms: Dict[str, Dict[WebSocket, Dict]] = {
    DEFAULT_ROOM: {}
}

async def translate(src_lang, tgt_lang, text, speaker_id):
    speaker_id = int(speaker_id)
    try:
        audio, sample_rate, translated_text = await t2S_translate(text, tgt_lang, speaker_id)
        if len(audio) == 0:
            print("WARNING: Empty audio received!")
            return translated_text, ""

        print(f"Final audio for WAV - shape: {audio.shape}, min/max: {audio.min()}/{audio.max()}")

        # Convert float32 normalized audio (-1.0 to 1.0) back to int16 PCM
        int16_audio = (audio * 32767).astype(np.int16)

        # Create WAV in memory buffer
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wav_file:
            n_channels = 1
            sampwidth = 2  # bytes for int16
            wav_file.setnchannels(n_channels)
            wav_file.setsampwidth(sampwidth)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(int16_audio.tobytes())

        buffer.seek(0)
        audio_b64 = base64.b64encode(buffer.read()).decode('utf-8')

        print(f"Base64 audio length: {len(audio_b64)} chars")
        return translated_text, audio_b64

    except Exception as e:
        print(f"Error in translate function: {e}")
        return f"Translation error: {str(e)}", ""


async def group_translate(connections, src_lang: str, tgt_lang: str, text: str, speaker_id: int):
    translated_text, audio_b64 = await translate(src_lang, tgt_lang, text, speaker_id)

    for ws in connections:
        try:
            await ws.send_json({
              "type" : "translate_msg",
              "transcript": text,
              "translated_text": translated_text,
              "translated_audio_url": audio_b64,
              "src_lang": src_lang,
              "tgt_lang": tgt_lang,
            })
            print(f"WebSocket x recieved src_lang: {src_lang}, tgt_lang: {tgt_lang}")
        except Exception as e:
          print(f"Error translating for group {tgt_lang}/{speaker_id}: {e}")

async def just_send(ws: WebSocket, src_lang: str, text: str):

    try:
        await ws.send_json({
            "type" : "transcript_msg",
            "transcript": text,
            "src_lang": src_lang
        })
        print(f"WebSocket x recieved src_lang: {src_lang}")
    except Exception as e:
        print(f"Error: {e}")

async def per_record(connections, per: bool):
    for ws in connections:
        try:
            await ws.send_json({
                "type" : "per_record",
                "per_record": per,
            })
            print(f"per_record: {per}")
        except Exception as e:
            print(f"Error: {e}")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    global current_recorder

    # Send current per_record state to the new user
    await websocket.send_json({
        "type": "per_record",
        "per_record": current_recorder is None  # True => اجازه رکورد هست
    })

    # Set default values
    user_data = {
        "lang": "en-US",
        "speaker_id": "0",
        "last_ping": time.time()
    }

    # Add user to default room
    rooms[DEFAULT_ROOM][websocket] = user_data
    last_active = time.time()

    try:
        while True:
            if websocket.client_state != WebSocketState.CONNECTED:
                break
            try:
                data = await websocket.receive_json()
                last_active = time.time()
                # global current_recorder

                if data.get("type") == "update_settings":
                    lang = data.get("lang", "en-US")
                    speaker_id = int(data.get("speaker_id", "0"))
                    rooms[DEFAULT_ROOM][websocket]["lang"] = lang
                    rooms[DEFAULT_ROOM][websocket]["speaker_id"] = speaker_id
                    await websocket.send_json({"status": "settings_updated"})
                    print(f"WebSocket {websocket} updated settings: lang={lang}, speaker_id={speaker_id}")

                elif data.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
                    user_data["last_ping"] = last_active

                elif data.get("type") == "speak":
                    src_lang = data.get("src_lang")
                    text = data.get("text")
                    speaker_id = int(data.get("speaker_id", "0"))
                    print(f"ws x , src_lang: {src_lang} *type speak")
                    if not src_lang or not text:
                        continue

                    # Group users by their language + speaker_id
                    groups = defaultdict(list)
                    for ws, info in rooms[DEFAULT_ROOM].items():
                        key = (info["lang"], info["speaker_id"])
                        groups[key].append(ws)

                    tasks = []
                    for (tgt_lang, speaker_id), connections in groups.items():
                        if src_lang == tgt_lang:
                            for ws in connections:
                                tasks.append(
                                    just_send(ws, src_lang, text)
                                    )
                        else:
                            tasks.append(
                                group_translate(connections, src_lang, tgt_lang, text, speaker_id)
                            )
                    await asyncio.gather(*tasks)
                elif data.get("type") == "status_Record":
                    if data.get("statusRecord") == True:
                        await per_record(list(rooms[DEFAULT_ROOM].keys()), False)
                        current_recorder = websocket
                    elif data.get("statusRecord") == False:
                        await per_record(list(rooms[DEFAULT_ROOM].keys()), True)
                        current_recorder = None


            except Exception as e:
                print(f"Client error: {e}")
                break

            if time.time() - last_active > PING_TIMEOUT:
                print("Client inactive, disconnecting.")
                break

    except Exception as e:
        print(f"Connection error: {e}")

    finally:
        rooms[DEFAULT_ROOM].pop(websocket, None)

        # همیشه بررسی کنیم که اگر این کاربر رکوردر بود، آن را خالی کنیم
        if current_recorder == websocket:
            await per_record(list(rooms[DEFAULT_ROOM].keys()), True)
            current_recorder = None

        # تلاش برای بستن سوکت
        try:
            await websocket.close()
        except Exception as e:
            print(f"Error closing WebSocket: {e}")


import uvicorn

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
