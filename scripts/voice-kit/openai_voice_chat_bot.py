#!/usr/bin/env python3
"""Compatibility launcher for the current Rust voice agent.

The default path execs `bin/gemma-agent-harness`, using the micro:bit A button
over USB serial, the USB webcam microphone, iPhone-hosted Gemma, iPhone Kokoro
TTS, and HDMI playback. Set GEMMA_LEGACY_OPENAI_VOICE_BOT=1 to run the older
AIY Voice HAT Python implementation below.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

if os.environ.get("GEMMA_LEGACY_OPENAI_VOICE_BOT") != "1":
    root = Path.home() / "gemma4-robot"
    harness = root / "bin" / "gemma-agent-harness"
    env_file = root / ".env"
    os.execv(
        str(harness),
        [
            str(harness),
            "--env-file",
            str(env_file),
            "voice-bot",
            "--button-source",
            "microbit-serial",
            "--microbit-device",
            "auto",
            "--led-source",
            "none",
            "--playback-device",
            "plughw:vc4hdmi,0",
            "--capture-device",
            "plughw:Camera,0",
            "--sample-rate",
            "48000",
            "--channels",
            "2",
            "--transcription-provider",
            "none",
            "--tts-provider",
            "iphone",
            "--iphone-tts-backend",
            "fluid-kokoro-ane",
            "--startup-greeting",
            "",
        ],
    )

import argparse
import json
import re
import signal
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from typing import Any

from gpiozero import Button, LED


DEFAULT_BASE_DIR = Path.home() / "gemma4-robot" / "voice-chat"
DEFAULT_ENV_FILE = Path.home() / "gemma4-robot" / ".env"
DEFAULT_STATUS_FILE = Path.home() / "gemma4-robot" / "kiosk" / "status.json"
DEFAULT_EXERCISE_STATUS_FILE = Path.home() / "gemma4-robot" / "kiosk" / "exercise_status.json"
DEFAULT_EXERCISE_FRAME_FILE = Path.home() / "gemma4-robot" / "kiosk" / "exercise_frame.rgb"
DEFAULT_EXERCISE_SCRIPT = Path.home() / "gemma4-robot" / "scripts" / "voice-kit" / "pose_preview_mode.py"
DEFAULT_EXERCISE_LOG = Path("/tmp/gemma4-pose-preview.log")
DEFAULT_POSE_RUNTIME = Path.home() / "gemma4-robot" / "out" / "pose_neon_runtime_aarch64_ofast"
DEFAULT_POSE_DATA_DIR = Path.home() / "gemma4-robot" / "out" / "pose_runtime_data"
VOICE_CARD_RE = re.compile(r"(google|voice|aiy|sndrpigoogle)", re.IGNORECASE)
CARD_LINE_RE = re.compile(
    r"^card\s+(?P<card>\d+):\s+(?P<short>[^\[]+)\[(?P<long>[^\]]+)\],"
    r"\s+device\s+(?P<device>\d+):\s+(?P<device_name>.+)$"
)


@dataclass(frozen=True)
class AlsaDevice:
    card: int
    device: int
    short: str
    long: str
    device_name: str

    @property
    def plughw(self) -> str:
        return f"plughw:{self.card},{self.device}"

    @property
    def searchable_text(self) -> str:
        return " ".join([self.short, self.long, self.device_name])


class OpenAIError(RuntimeError):
    pass


def run_text(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as exc:
        return exc.output


def list_alsa_devices(command: str) -> list[AlsaDevice]:
    output = run_text([command, "-l"])
    devices: list[AlsaDevice] = []
    for line in output.splitlines():
        match = CARD_LINE_RE.match(line.strip())
        if not match:
            continue
        devices.append(
            AlsaDevice(
                card=int(match.group("card")),
                device=int(match.group("device")),
                short=match.group("short").strip(),
                long=match.group("long").strip(),
                device_name=match.group("device_name").strip(),
            )
        )
    return devices


def choose_alsa_device(kind: str, override: str | None) -> str:
    if override:
        return override

    command = "aplay" if kind == "playback" else "arecord"
    devices = list_alsa_devices(command)
    for device in devices:
        if VOICE_CARD_RE.search(device.searchable_text):
            return device.plughw

    if kind == "capture" and len(devices) == 1:
        return devices[0].plughw

    choices = ", ".join(device.plughw for device in devices) or "none"
    raise RuntimeError(f"No AIY Voice HAT {kind} device found; available: {choices}")


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def require_api_key(env_file: Path) -> str:
    load_env_file(env_file)
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError(f"OPENAI_API_KEY is not set and was not found in {env_file}")
    return key


def openai_request(api_key: str, path: str, body: bytes, content_type: str) -> bytes:
    request = urllib.request.Request(
        f"https://api.openai.com/v1/{path.lstrip('/')}",
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": content_type,
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise OpenAIError(f"OpenAI API error {exc.code}: {error_body}") from exc
    except urllib.error.URLError as exc:
        raise OpenAIError(f"OpenAI API request failed: {exc}") from exc


def openai_json(api_key: str, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    data = openai_request(api_key, path, body, "application/json")
    return json.loads(data.decode("utf-8"))


def multipart_form(parts: dict[str, str], file_field: str, file_path: Path) -> tuple[bytes, str]:
    boundary = f"----gemma4robot{uuid.uuid4().hex}"
    chunks: list[bytes] = []
    for name, value in parts.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode("utf-8"),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"),
                value.encode("utf-8"),
                b"\r\n",
            ]
        )

    chunks.extend(
        [
            f"--{boundary}\r\n".encode("utf-8"),
            (
                f'Content-Disposition: form-data; name="{file_field}"; '
                f'filename="{file_path.name}"\r\n'
            ).encode("utf-8"),
            b"Content-Type: audio/wav\r\n\r\n",
            file_path.read_bytes(),
            b"\r\n",
            f"--{boundary}--\r\n".encode("utf-8"),
        ]
    )
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def extract_response_text(response: dict[str, Any]) -> str:
    if isinstance(response.get("output_text"), str) and response["output_text"].strip():
        return response["output_text"].strip()

    texts: list[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            if value.get("type") == "output_text" and isinstance(value.get("text"), str):
                texts.append(value["text"])
            else:
                for item in value.values():
                    walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(response.get("output", []))
    return "\n".join(part.strip() for part in texts if part.strip()).strip()


def split_for_tts(text: str, max_chars: int) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return [text] if text else []

    chunks: list[str] = []
    current = ""
    for piece in re.split(r"(?<=[.!?])\s+", text):
        if not piece:
            continue
        if current and len(current) + 1 + len(piece) > max_chars:
            chunks.append(current)
            current = piece
        else:
            current = piece if not current else f"{current} {piece}"
    if current:
        chunks.append(current)

    final: list[str] = []
    for chunk in chunks:
        while len(chunk) > max_chars:
            final.append(chunk[:max_chars])
            chunk = chunk[max_chars:]
        if chunk:
            final.append(chunk)
    return final


class VoiceChatBot:
    def __init__(self, args: argparse.Namespace, api_key: str, playback: str, capture: str) -> None:
        self.args = args
        self.api_key = api_key
        self.playback = playback
        self.capture = capture
        self.base_dir = Path(args.base_dir).expanduser()
        self.recordings_dir = self.base_dir / "recordings"
        self.speech_dir = self.base_dir / "speech"
        self.status_file = Path(args.status_file).expanduser()
        self.exercise_status_file = Path(args.exercise_status_file).expanduser()
        self.exercise_frame_file = Path(args.exercise_frame_file).expanduser()
        self.exercise_script = Path(args.exercise_script).expanduser()
        self.exercise_log = Path(args.exercise_log).expanduser()
        self.recordings_dir.mkdir(parents=True, exist_ok=True)
        self.speech_dir.mkdir(parents=True, exist_ok=True)
        self.status_file.parent.mkdir(parents=True, exist_ok=True)
        self.exercise_status_file.parent.mkdir(parents=True, exist_ok=True)

        self.led = LED(args.led_gpio)
        self.lock = threading.RLock()
        self.recording_proc: subprocess.Popen[str] | None = None
        self.recording_path: Path | None = None
        self.recording_started = 0.0
        self.exercise_proc: subprocess.Popen[str] | None = None
        self.exercise_log_handle: Any | None = None
        self.button_is_pressed = False
        self.button_pressed_at = 0.0
        self.second_click_active = False
        self.pending_tap_timer: threading.Timer | None = None
        self.recording_start_timer: threading.Timer | None = None
        self.previous_response_id: str | None = None
        self.turn_index = 0
        self.latest_input = ""
        self.latest_output = ""
        self.latest_error = ""
        self.write_status("idle")

    def write_status(self, state: str) -> None:
        payload = {
            "mode": "camera" if state == "camera" else "voice",
            "state": state,
            "turn": self.turn_index,
            "input": self.latest_input,
            "output": self.latest_output,
            "error": self.latest_error,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        }
        tmp = self.status_file.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        tmp.replace(self.status_file)

    def reset_conversation(self, reason: str) -> None:
        with self.lock:
            self.previous_response_id = None
            self.turn_index = 0
            self.latest_input = ""
            self.latest_output = ""
            self.latest_error = ""
            self.write_status("reset")
        print(f"Conversation reset: {reason}", flush=True)

    def set_ready_light(self) -> None:
        self.led.off()

    def set_recording_light(self) -> None:
        self.led.on()

    def set_waiting_light(self) -> None:
        self.led.blink(on_time=0.25, off_time=0.25, background=True)

    def set_speaking_light(self) -> None:
        self.led.on()

    def start_recording(self) -> None:
        with self.lock:
            self._start_recording_locked()

    def _start_recording_locked(self) -> None:
        if self.exercise_running_locked():
            print("Camera preview is active; ignoring voice recording press.", flush=True)
            return
        if self.recording_proc is not None:
            print("Already recording; ignoring press.", flush=True)
            return
        self.turn_index += 1
        stamp = time.strftime("%Y%m%d-%H%M%S")
        path = self.recordings_dir / f"turn-{self.turn_index:03d}-{stamp}.wav"
        command = [
            "arecord",
            "-q",
            "-D",
            self.capture,
            "-f",
            "S16_LE",
            "-r",
            str(self.args.sample_rate),
            "-c",
            str(self.args.channels),
            str(path),
        ]
        self.set_recording_light()
        self.recording_path = path
        self.recording_started = time.monotonic()
        self.recording_proc = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        self.latest_error = ""
        self.write_status("recording")
        print(f"Recording to {path}", flush=True)

    def exercise_running_locked(self) -> bool:
        return self.exercise_proc is not None and self.exercise_proc.poll() is None

    def cancel_pending_tap_locked(self) -> None:
        if self.pending_tap_timer is not None:
            self.pending_tap_timer.cancel()
            self.pending_tap_timer = None

    def cancel_recording_start_locked(self) -> None:
        if self.recording_start_timer is not None:
            self.recording_start_timer.cancel()
            self.recording_start_timer = None

    def schedule_recording_start_locked(self) -> None:
        self.cancel_recording_start_locked()
        timer = threading.Timer(self.args.hold_start_seconds, self.start_recording_if_button_still_down)
        timer.daemon = True
        self.recording_start_timer = timer
        timer.start()

    def start_recording_if_button_still_down(self) -> None:
        with self.lock:
            self.recording_start_timer = None
            if not self.button_is_pressed or self.second_click_active:
                return
            self._start_recording_locked()

    def schedule_single_tap_locked(self) -> None:
        self.cancel_pending_tap_locked()
        timer = threading.Timer(self.args.double_click_seconds, self.handle_single_tap_timeout)
        timer.daemon = True
        self.pending_tap_timer = timer
        timer.start()

    def handle_single_tap_timeout(self) -> None:
        with self.lock:
            self.pending_tap_timer = None
            if self.exercise_running_locked():
                return
        self.set_ready_light()
        self.reset_conversation(f"single tap shorter than {self.args.tap_reset_seconds:.2f}s")

    def handle_button_pressed(self) -> None:
        now = time.monotonic()
        with self.lock:
            self.button_is_pressed = True
            self.button_pressed_at = now
            if self.pending_tap_timer is not None:
                self.cancel_pending_tap_locked()
                self.cancel_recording_start_locked()
                self.second_click_active = True
                return

            self.second_click_active = False
            if not self.exercise_running_locked():
                self.schedule_recording_start_locked()

    def handle_button_released(self) -> None:
        now = time.monotonic()
        should_toggle_exercise = False
        should_stop_recording = False
        with self.lock:
            held = max(0.0, now - self.button_pressed_at)
            self.button_is_pressed = False
            self.cancel_recording_start_locked()

            if self.second_click_active:
                should_toggle_exercise = held <= self.args.double_click_press_seconds
                self.second_click_active = False
            elif self.recording_proc is not None:
                should_stop_recording = True
            elif held < self.args.tap_reset_seconds:
                self.schedule_single_tap_locked()

        if should_toggle_exercise:
            threading.Thread(target=self.toggle_exercise_mode, daemon=True).start()
        elif should_stop_recording:
            self.stop_recording()

    def terminate_current_recording(self) -> None:
        with self.lock:
            proc = self.recording_proc
            self.recording_proc = None
            self.recording_path = None
        if proc is None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=2)

    def exercise_command(self) -> list[str]:
        return [
            sys.executable,
            str(self.exercise_script),
            "--pose-runtime",
            str(Path(self.args.pose_runtime).expanduser()),
            "--pose-data-dir",
            str(Path(self.args.pose_data_dir).expanduser()),
            "--width",
            str(self.args.exercise_width),
            "--height",
            str(self.args.exercise_height),
            "--framerate",
            str(self.args.camera_framerate),
            "--roi",
            self.args.camera_roi,
            "--threads",
            str(self.args.exercise_threads),
            "--cores",
            self.args.exercise_cores,
            "--latency-log",
            str(self.args.pose_latency_log),
        ]

    def stop_kiosk_processes(self) -> None:
        uid = str(os.getuid())
        patterns = [
            "chromium.*chromium-kiosk-profile",
            "openbox",
            "Xorg :0",
            "startx /tmp/gemma4-kiosk-xinitrc",
            "python3 -m http.server 8765",
        ]
        for signal_name in ["-TERM", "-KILL"]:
            for pattern in patterns:
                subprocess.run(
                    ["pkill", signal_name, "-u", uid, "-f", pattern],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
            if signal_name == "-TERM":
                time.sleep(1.0)

    def pose_exercise_command(self) -> list[str]:
        return [
            sys.executable,
            str(self.exercise_script),
            "--status-file",
            str(self.exercise_status_file),
            "--frame-file",
            str(self.exercise_frame_file),
            "--pose-runtime",
            str(Path(self.args.pose_runtime).expanduser()),
            "--pose-data-dir",
            str(Path(self.args.pose_data_dir).expanduser()),
            "--width",
            str(self.args.exercise_width),
            "--height",
            str(self.args.exercise_height),
            "--threads",
            str(self.args.exercise_threads),
        ]

    def start_exercise_mode(self) -> None:
        self.terminate_current_recording()
        self.stop_kiosk_processes()
        with self.lock:
            if self.exercise_running_locked():
                return
            self.cancel_pending_tap_locked()
            self.cancel_recording_start_locked()
            self.exercise_log.parent.mkdir(parents=True, exist_ok=True)
            self.exercise_log_handle = self.exercise_log.open("a", buffering=1)
            command = self.exercise_command()
            self.exercise_proc = subprocess.Popen(
                command,
                stdout=self.exercise_log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            self.latest_input = "Pose preview"
            self.latest_output = "Camera with pose skeleton overlay"
            self.latest_error = ""
            self.write_status("camera")
            self.led.blink(on_time=0.12, off_time=0.88, background=True)
            print(f"Pose preview started with pid {self.exercise_proc.pid}; log={self.exercise_log}", flush=True)
            threading.Thread(
                target=self.monitor_exercise_process,
                args=(self.exercise_proc,),
                daemon=True,
            ).start()

    def monitor_exercise_process(self, proc: subprocess.Popen[str]) -> None:
        rc = proc.wait()
        with self.lock:
            if self.exercise_proc is not proc:
                return
            log_handle = self.exercise_log_handle
            self.exercise_proc = None
            self.exercise_log_handle = None
            self.latest_input = ""
            self.latest_output = ""
            self.latest_error = "" if rc == 0 else f"Pose preview exited with {rc}."
            self.write_status("idle" if rc == 0 else "error")
            self.set_ready_light()
        if log_handle is not None:
            log_handle.close()
        print(f"Pose preview process exited with {rc}.", flush=True)

    def stop_exercise_mode(self) -> None:
        with self.lock:
            proc = self.exercise_proc
            log_handle = self.exercise_log_handle
            self.exercise_proc = None
            self.exercise_log_handle = None
            self.latest_input = ""
            self.latest_output = ""
            self.latest_error = ""
            self.write_status("idle")
            self.set_ready_light()

        if proc is not None and proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait(timeout=2)
        if log_handle is not None:
            log_handle.close()
        print("Pose preview stopped; voice assistant ready.", flush=True)

    def toggle_exercise_mode(self) -> None:
        with self.lock:
            running = self.exercise_running_locked()
        if running:
            self.stop_exercise_mode()
        else:
            self.start_exercise_mode()

    def stop_recording(self) -> None:
        with self.lock:
            proc = self.recording_proc
            path = self.recording_path
            started = self.recording_started
            self.recording_proc = None
            self.recording_path = None

        if proc is None or path is None:
            return

        duration = max(0.0, time.monotonic() - started)
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=2)

        if duration < self.args.tap_reset_seconds:
            self.set_ready_light()
            self.reset_conversation(f"tap shorter than {self.args.tap_reset_seconds:.2f}s")
            return

        if not path.exists() or path.stat().st_size <= 44:
            self.set_ready_light()
            self.latest_error = "Recording was empty."
            self.write_status("idle")
            print(f"Recording was empty; arecord exited with {proc.returncode}; skipping turn.", flush=True)
            return

        if proc.returncode not in (0, -signal.SIGTERM, 1, 143):
            print(f"arecord exited with {proc.returncode}, but WAV data exists; processing it.", flush=True)

        self.set_waiting_light()
        self.write_status("processing")
        threading.Thread(target=self.process_turn, args=(path, duration), daemon=True).start()

    def transcribe(self, wav_path: Path) -> str:
        parts = {
            "model": self.args.transcription_model,
            "response_format": "text",
        }
        if self.args.language:
            parts["language"] = self.args.language
        body, content_type = multipart_form(parts, "file", wav_path)
        data = openai_request(self.api_key, "audio/transcriptions", body, content_type)
        return data.decode("utf-8", errors="replace").strip()

    def ask_model(self, transcript: str) -> str:
        payload: dict[str, Any] = {
            "model": self.args.response_model,
            "instructions": self.args.instructions,
            "input": transcript,
            "max_output_tokens": self.args.max_output_tokens,
            "truncation": "auto",
        }
        with self.lock:
            if self.previous_response_id:
                payload["previous_response_id"] = self.previous_response_id

        response = openai_json(self.api_key, "responses", payload)
        text = extract_response_text(response)
        if not text:
            raise OpenAIError(f"Responses API returned no output text: {json.dumps(response)[:1000]}")

        response_id = response.get("id")
        if isinstance(response_id, str) and response_id:
            with self.lock:
                self.previous_response_id = response_id
        return text

    def synthesize(self, text: str, path: Path) -> None:
        payload: dict[str, Any] = {
            "model": self.args.tts_model,
            "voice": self.args.voice,
            "input": text,
            "response_format": "wav",
        }
        if self.args.tts_instructions:
            payload["instructions"] = self.args.tts_instructions
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        audio = openai_request(self.api_key, "audio/speech", body, "application/json")
        path.write_bytes(audio)

    def play_wav(self, path: Path) -> None:
        subprocess.run(["aplay", "-q", "-D", self.playback, str(path)], check=True)

    def speak(self, text: str, name_prefix: str) -> None:
        chunks = split_for_tts(text, self.args.tts_max_chars)
        for index, chunk in enumerate(chunks, start=1):
            path = self.speech_dir / f"{name_prefix}-{index:02d}.wav"
            self.synthesize(chunk, path)
            self.play_wav(path)

    def process_turn(self, wav_path: Path, duration: float) -> None:
        try:
            print(f"Stopped recording after {duration:.1f}s; transcribing.", flush=True)
            transcript = self.transcribe(wav_path)
            if not transcript:
                print("Whisper returned an empty transcript; ready.", flush=True)
                self.latest_error = "Whisper returned an empty transcript."
                self.write_status("idle")
                self.set_ready_light()
                return

            self.latest_input = transcript
            self.latest_output = ""
            self.latest_error = ""
            self.write_status("thinking")
            print(f"User: {transcript}", flush=True)
            answer = self.ask_model(transcript)
            self.latest_output = answer
            self.write_status("speaking")
            print(f"Assistant: {answer}", flush=True)
            self.set_speaking_light()
            self.speak(answer, f"turn-{self.turn_index:03d}")
            self.write_status("idle")
            print("Ready.", flush=True)
        except Exception as exc:
            self.latest_error = str(exc)
            self.write_status("error")
            print(f"Turn failed: {exc}", file=sys.stderr, flush=True)
        finally:
            self.set_ready_light()

    def speak_startup_greeting(self) -> None:
        if not self.args.startup_greeting:
            return
        try:
            self.set_speaking_light()
            self.speak(self.args.startup_greeting, "startup")
        except Exception as exc:
            print(f"Startup greeting failed: {exc}", file=sys.stderr, flush=True)
        finally:
            self.set_ready_light()

    def run_self_test(self) -> None:
        self.set_waiting_light()
        synthetic_input = self.speech_dir / "self-test-input.wav"
        self.synthesize("Testing one two three.", synthetic_input)
        transcript = self.transcribe(synthetic_input)
        print(f"Self-test transcript: {transcript}", flush=True)
        answer = self.ask_model("Reply with exactly this sentence: Voice chat test is working.")
        print(f"Self-test assistant: {answer}", flush=True)
        self.set_speaking_light()
        self.speak(answer, "self-test-output")
        self.set_ready_light()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_FILE), help="file containing OPENAI_API_KEY")
    parser.add_argument("--base-dir", default=str(DEFAULT_BASE_DIR), help="directory for recordings and speech")
    parser.add_argument("--status-file", default=str(DEFAULT_STATUS_FILE), help="JSON status file for kiosk display")
    parser.add_argument("--exercise-status-file", default=str(DEFAULT_EXERCISE_STATUS_FILE), help="exercise JSON status file for kiosk display")
    parser.add_argument("--exercise-frame-file", default=str(DEFAULT_EXERCISE_FRAME_FILE), help="raw RGB camera frame for kiosk display")
    parser.add_argument("--exercise-script", default=str(DEFAULT_EXERCISE_SCRIPT), help="pose preview runner")
    parser.add_argument("--exercise-log", default=str(DEFAULT_EXERCISE_LOG), help="pose preview process log file")
    parser.add_argument("--pose-runtime", default=str(DEFAULT_POSE_RUNTIME), help="pose runtime binary used by exercise mode")
    parser.add_argument("--pose-data-dir", default=str(DEFAULT_POSE_DATA_DIR), help="pose runtime data directory")
    parser.add_argument("--exercise-width", type=int, default=320, help="exercise camera frame width")
    parser.add_argument("--exercise-height", type=int, default=240, help="exercise camera frame height")
    parser.add_argument("--exercise-threads", type=int, default=2, help="pose runtime worker threads for pose preview")
    parser.add_argument("--exercise-cores", default="0,1", help="CPU cores allowed for pose runtime")
    parser.add_argument("--pose-latency-log", default="/tmp/gemma4-pose-preview-latencies.jsonl", help="pose preview latency JSONL log")
    parser.add_argument("--camera-preview-command", default="rpicam-hello", help="command used for direct HDMI camera preview")
    parser.add_argument("--camera-width", type=int, default=1280, help="direct preview stream width")
    parser.add_argument("--camera-height", type=int, default=720, help="direct preview stream height")
    parser.add_argument("--camera-framerate", type=float, default=8.0, help="pose preview camera stream FPS")
    parser.add_argument("--camera-mode", default="", help="optional rpicam camera mode")
    parser.add_argument("--camera-info-text", default="", help="optional rpicam preview overlay text")
    parser.add_argument("--camera-roi", default="", help="optional center crop for pose preview; empty keeps maximum field of view")
    parser.add_argument("--button-gpio", type=int, default=23, help="BCM GPIO for the AIY button")
    parser.add_argument("--led-gpio", type=int, default=25, help="BCM GPIO for the AIY button LED")
    parser.add_argument("--playback-device", help="explicit ALSA playback device, e.g. plughw:1,0")
    parser.add_argument("--capture-device", help="explicit ALSA capture device, e.g. plughw:1,0")
    parser.add_argument("--sample-rate", type=int, default=16000, help="recording sample rate")
    parser.add_argument("--channels", type=int, default=1, help="recording channel count")
    parser.add_argument("--tap-reset-seconds", type=float, default=0.35, help="tap shorter than this resets chat")
    parser.add_argument("--double-click-seconds", type=float, default=0.55, help="max gap between short clicks for pose preview toggle")
    parser.add_argument("--double-click-press-seconds", type=float, default=0.45, help="max duration for the second pose-preview click")
    parser.add_argument("--hold-start-seconds", type=float, default=0.28, help="delay before push-to-talk recording starts")
    parser.add_argument("--response-model", default="gpt-5.5", help="Responses API model")
    parser.add_argument("--transcription-model", default="whisper-1", help="audio transcription model")
    parser.add_argument("--tts-model", default="gpt-4o-mini-tts", help="speech model")
    parser.add_argument("--voice", default="alloy", help="TTS voice")
    parser.add_argument("--language", help="optional ISO-639-1 hint for Whisper, e.g. en")
    parser.add_argument("--max-output-tokens", type=int, default=500, help="maximum response tokens")
    parser.add_argument("--tts-max-chars", type=int, default=3900, help="max chars per TTS request")
    parser.add_argument(
        "--instructions",
        default=(
            "You are a helpful voice assistant running on a Raspberry Pi robot. "
            "Answer conversationally and keep replies concise unless the user asks for detail."
        ),
        help="Responses API instructions",
    )
    parser.add_argument(
        "--tts-instructions",
        default="Speak naturally, clearly, and warmly. Keep a steady pace.",
        help="voice style instructions for compatible TTS models",
    )
    parser.add_argument(
        "--startup-greeting",
        default="",
        help="spoken greeting on startup; set to empty string to disable",
    )
    parser.add_argument("--once-self-test", action="store_true", help="test OpenAI TTS playback, then exit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env_file = Path(args.env_file).expanduser()
    api_key = require_api_key(env_file)
    playback = choose_alsa_device("playback", args.playback_device)
    capture = choose_alsa_device("capture", args.capture_device)
    print(f"Using playback={playback} capture={capture}", flush=True)

    bot = VoiceChatBot(args, api_key, playback, capture)
    if args.once_self_test:
        bot.run_self_test()
        return

    button = Button(args.button_gpio, pull_up=True, bounce_time=0.05)
    button.when_pressed = bot.handle_button_pressed
    button.when_released = bot.handle_button_released

    bot.speak_startup_greeting()
    print(
        f"Ready. Hold the button on BCM GPIO {args.button_gpio} to talk; "
        f"tap shorter than {args.tap_reset_seconds:.2f}s to reset; "
        "double-click to toggle pose preview.",
        flush=True,
    )

    def shutdown(_signum: int, _frame: object) -> None:
        bot.stop_exercise_mode()
        raise SystemExit(0)

    def toggle_camera_preview(_signum: int, _frame: object) -> None:
        threading.Thread(target=bot.toggle_exercise_mode, daemon=True).start()

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGUSR1, toggle_camera_preview)
    while True:
        signal.pause()


if __name__ == "__main__":
    main()
