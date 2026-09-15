import os
import time
import wave
import json
import queue
import socket
import struct
import threading
import tempfile

import torch
import pyaudio
import requests
import azure.cognitiveservices.speech as speechsdk

#from SpeakerRecognizer import SpeakerRecognizer

# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------
FINAL_SILENCE_TIME = 2.0    # seconds of silence before closing a turn
SPLIT_TIME         = 0.5    # seconds of silence before closing a VAD chunk
MAX_CHUNK_SEC      = 6.0   # hard cap on chunk length to bound Azure latency
ENERGY_THRESHOLD   = 500    # RMS value to consider a frame as speech
VAD_THRESHOLD      = 0.7    # Silero VAD speech probability threshold (0-1)
RATE               = 16000  # Hz
CHUNK              = 512    # pyaudio frames per buffer — Silero VAD requires 512 samples at 16kHz
CHANNELS           = 1
SAMPLE_WIDTH       = 2      # bytes  (int16)
SOCKET_PORT        = 9090
# ---------------------------------------------------------------------------



def _write_wav(frames: list, path: str) -> None:
    """Write a list of raw PCM byte buffers to a WAV file."""
    with wave.open(path, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(RATE)
        wf.writeframes(b"".join(frames))


def _wav_duration(path: str) -> float:
    """Return duration of a WAV file in seconds."""
    with wave.open(path, "r") as wf:
        return wf.getnframes() / float(wf.getframerate())

def _load_silero_vad():
    """Download (first run) and return the Silero VAD model + get_speech_ts utility."""
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        onnx=False,
        verbose=False,
    )
    return model

class _ChunkResult:
    """
    Holds partial results for one recorded audio chunk.
    Both the speech thread and the speaker thread call set_speech / set_speaker;
    when both have reported the internal Event is set and wait() unblocks.
    """

    def __init__(self, chunk_id: int, enable_speaker: bool = True):
        self.chunk_id = chunk_id
        self.enable_speaker = enable_speaker
        self.text: str    = ""
        self.speaker: str = "unknown"
        self.score: float = 0.0
        self._lock        = threading.Lock()
        self._done        = threading.Event()
        self._parts_done  = 0          # increments 0 → 1 → 2

        # ── timing ────────────────────────────────────────────────────────────
        self.t_speech_start:  float = 0.0   # first VAD-positive frame
        self.t_flushed:       float = 0.0   # WAV written, dispatched
        self.t_azure_start:   float = 0.0   # recognize_once() called
        self.t_azure_end:     float = 0.0   # recognize_once() returned
        self.t_speaker_start: float = 0.0   # identify_from_wav() called
        self.t_speaker_end:   float = 0.0   # identify_from_wav() returned
        self.t_collected:     float = 0.0   # collector finished

    def set_speech(self, text: str):
        with self._lock:
            self.text = text
            if not self.enable_speaker:
                self._done.set()
            else:
                self._parts_done += 1
                if self._parts_done == 2:
                    self._done.set()

    def set_speaker(self, speaker: str, score: float):
        with self._lock:
            self.speaker = speaker
            self.score   = score
            self._parts_done += 1
            if self._parts_done == 2:
                self._done.set()

    def wait(self, timeout: float = 15.0) -> bool:
        return self._done.wait(timeout=timeout)


class Recognizer:
    """
    VAD-segmented speech + speaker recognizer.

    Architecture
    ------------
    _mic_thread
        Reads raw PCM from the microphone.
        Silero VAD delimits speech segments:
          • while user speaking  → accumulate frames
          • after SPLIT_TIME of silence      → flush segment to disk as WAV
                                               and post to the pipeline
        While self.speaking is True, audio is discarded (no Azure interaction).

    _dispatch_thread
        Picks up (_ChunkResult, wav_path) pairs and spawns two daemon threads
        per chunk:
            _speech_thread  → Azure recognize_once() on the WAV file
            _speaker_thread → SpeakerRecognizer.identify_from_wav()

    _collector_thread
        Waits for _ChunkResult objects to complete (in integer ID order) and
        appends valid, non-robot results to self.detected_speech.

    listen() / simple_listen()
        Main turn-taking loops.  After FINAL_SILENCE_TIME with no new chunks,
        recognized_something() assembles the turn and the result is sent
        over the socket wrapped in XML.
    """

    def __init__(self, language: str, azure_key: str, azure_region: str,
                 speaker_threshold: float = 0.42,
                 emoACT_ip: str = "127.0.0.1",
                 enable_speaker_recognition: bool = True, 
                 emoACT_active: bool = True):

        # ---- Azure credentials ----
        self._language     = language
        self._azure_key    = azure_key
        self._azure_region = azure_region

        # ---- speaker recognition enabled ----
        self.enable_speaker_recognition = enable_speaker_recognition
        self.emoACT_active = emoACT_active
        # ---- speaker recognizer ----
        if enable_speaker_recognition:
            self.speaker_recognizer = SpeakerRecognizer(threshold=speaker_threshold)
        else:
            self.speaker_recognizer = None

        # ---- assembled results ----
        self.detected_speech: list = []   # list of {"speaker": …, "text": …}
        self.recognized_text  = ""
        self.final_speaker    = "unknown"

        # ---- control flags ----
        self.running  = False
        self.speaking = False             # True while robot is speaking

        # ---- turn-level timing ----
        self.t_last_chunk_collected: float = 0.0   # when collector finished last chunk of a turn
        self.t_final = 0.0                            # when FINAL_SILENCE should expire

        # ---- chunk ID counter ----
        self._next_chunk_id = 0
        self._chunk_id_lock = threading.Lock()

        # ---- inter-thread queues ----
        # mic → dispatch: (ChunkResult, wav_path)
        self._dispatch_queue: queue.Queue = queue.Queue()
        # mic → collector:  ChunkResult  (in recording order)
        self._collect_queue:  queue.Queue = queue.Queue()

        # ---- EmoACT ----
        self.url_emoACT = f"http://{emoACT_ip}:4000/"

        # ---- pause-mode local recognition ----
        self.pause_mode = False
        self._pause_vosk_model = None
        self._pause_vosk_recognizer_cls = None
        self._pause_model_path = "model"

        # ---- temp dir for WAV chunks ----
        self._tmp_dir = tempfile.mkdtemp(prefix="sst_chunks_")
        print(f"[SST] Temp WAV dir: {self._tmp_dir}")

        # ---- Silero VAD model ----
        print("[SST] Loading Silero VAD model…")
        self._vad_model = _load_silero_vad()
        print("[SST] Silero VAD ready.")

        # ---- socket server ----
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind(("0.0.0.0", SOCKET_PORT))
        self.server.listen(5)

    # =========================================================================
    # Start / Stop / Close
    # =========================================================================

    def start(self):
        if self.running:
            return
        print("[SST] Starting threads…")
        self.running = True
        threading.Thread(target=self._mic_thread,        daemon=True).start()
        threading.Thread(target=self._dispatch_thread,   daemon=True).start()
        threading.Thread(target=self._collector_thread,  daemon=True).start()

    def stop(self):
        print("[SST] Stopping…")
        self.running = False

    def close(self):
        self.stop()
        try:
            self.server.shutdown(socket.SHUT_RDWR)
        except Exception:
            pass
        try:
            self.server.close()
        except Exception:
            pass
        print("[SST] Closed.")

    def set_speaking(self, s: bool):
        """Call True when robot starts speaking, False when it finishes."""
        print(f"[SST] set_speaking({s})")
        self.speaking = s

    def _ensure_pause_model(self) -> bool:
        if self._pause_vosk_model is not None and self._pause_vosk_recognizer_cls is not None:
            return True
        try:
            from vosk import Model, KaldiRecognizer
        except ImportError as e:
            print("[SST] VOSK is not installed; pause-mode local recognition unavailable:", e)
            return False
        try:
            self._pause_vosk_model = Model(self._pause_model_path)
            self._pause_vosk_recognizer_cls = KaldiRecognizer
            print(f"[SST] Loaded VOSK pause model from {self._pause_model_path}")
            return True
        except Exception as e:
            print("[SST] Failed to load VOSK pause model:", e)
            self._pause_vosk_model = None
            self._pause_vosk_recognizer_cls = None
            return False

    def set_pause_mode(self, enabled: bool):
        self.pause_mode = bool(enabled)
        if self.pause_mode and not self._ensure_pause_model():
            print("[SST] Pause mode requested but local model unavailable; recognized text will be empty.")
        print(f"[SST] Pause mode {'enabled' if self.pause_mode else 'disabled'}")

    def _local_recognize(self, wav_path: str) -> str:
        if not self._ensure_pause_model():
            return ""
        recognizer = self._pause_vosk_recognizer_cls(self._pause_vosk_model, RATE)
        text = ""
        try:
            with wave.open(wav_path, 'rb') as wf:
                while True:
                    data = wf.readframes(4000)
                    if len(data) == 0:
                        break
                    recognizer.AcceptWaveform(data)
            result = json.loads(recognizer.FinalResult())
            text = result.get("text", "").strip()
        except Exception as e:
            print("[SST] Pause local recognition error:", e)
            text = ""
        return text

    # =========================================================================
    # Mic thread — VAD segmentation
    # =========================================================================

    def _mic_thread(self):
        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK
        )
        print("[Mic] Recording started.")
 
        segment_frames: list = []
        speech_detected      = False
        silence_start        = None        # wall-clock time silence began
        chunk_start          = None        # wall-clock time current segment started

        # Silero VAD state — model is stateful, reset between segments
        self._vad_model.reset_states()

        while self.running:
            data = stream.read(CHUNK, exception_on_overflow=False)
 
            # While the robot is talking: discard audio, reset state
            if self.speaking:
                segment_frames.clear()
                speech_detected = False
                silence_start   = None
                chunk_start     = None
                self._vad_model.reset_states()
                continue
 
            #is_speech = _rms(data) > ENERGY_THRESHOLD
            # ── Silero VAD ────────────────────────────────────────────────────
            # Convert 512-sample int16 PCM buffer to float32 tensor [-1, 1]
            audio_tensor = torch.frombuffer(bytearray(data), dtype=torch.int16).float() / 32768.0
            with torch.no_grad():
                speech_prob = self._vad_model(audio_tensor, RATE).item()
            is_speech = speech_prob >= VAD_THRESHOLD
 
            if is_speech:
                if chunk_start is None:
                    chunk_start = time.time()   # mark when this segment started
                segment_frames.append(data)
                self.t_final = time.time() + FINAL_SILENCE_TIME
                speech_detected = True
                silence_start   = None          # voice resumed, reset timer
            else:
                if speech_detected:
                    segment_frames.append(data)  # keep trailing silence frames
                    self.t_final = time.time() + FINAL_SILENCE_TIME  #update final silence deadline
                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start >= SPLIT_TIME:
                        # End of a VAD chunk → flush
                        self._flush_segment(list(segment_frames), chunk_start)
                        segment_frames.clear()
                        speech_detected = False
                        silence_start   = None
                        chunk_start     = None
                        continue
 
            # Hard cap: force-flush if chunk has grown too long,
            # even if the speaker hasn't paused. This keeps Azure recognize_once() latency bounded to ~MAX_CHUNK_SEC.
            if (speech_detected and chunk_start is not None
                    and time.time() - chunk_start >= MAX_CHUNK_SEC):
                self._flush_segment(list(segment_frames), chunk_start)
                segment_frames.clear()
                speech_detected = False
                silence_start   = None
                chunk_start     = None
 
        stream.stop_stream()
        stream.close()
        p.terminate()
        print("[Mic] Recording stopped.")

    def _flush_segment(self, frames: list, t_speech_start: float):
        """Write frames to a temp WAV and post it to both pipeline queues."""
        with self._chunk_id_lock:
            cid = self._next_chunk_id
            self._next_chunk_id += 1

        wav_path = os.path.join(self._tmp_dir, f"chunk_{cid:06d}.wav")
        _write_wav(frames, wav_path)
        duration = _wav_duration(wav_path)
        t_flushed = time.time()
    
        result = _ChunkResult(cid, enable_speaker=self.enable_speaker_recognition)
        result.t_speech_start = t_speech_start
        result.t_flushed      = t_flushed
        result.wav_duration     = duration
        
        print(f"[Mic] Flushed chunk {cid} ({duration:.2f}s)")
        self._dispatch_queue.put((result, wav_path))
        self._collect_queue.put(result)

    # =========================================================================
    # Dispatch thread — spawn speech + speaker threads per chunk
    # =========================================================================

    def _dispatch_thread(self):
        while self.running:
            try:
                result, wav_path = self._dispatch_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            threading.Thread(
                target=self._speech_thread,
                args=(result, wav_path),
                daemon=True
            ).start()
            if self.enable_speaker_recognition:
                threading.Thread(
                    target=self._speaker_thread,
                    args=(result, wav_path),
                    daemon=True
                ).start()

    # =========================================================================
    # Per-chunk: Azure speech recognition (recognize_once on WAV file)
    # =========================================================================

    def _speech_thread(self, result: _ChunkResult, wav_path: str):
        result.t_azure_start = time.time()
        if self.pause_mode:
            text = self._local_recognize(wav_path)
            print(f"[Speech {result.chunk_id} PAUSE] '{text}'")
        else:
            try:
                speech_config = speechsdk.SpeechConfig(
                    subscription=self._azure_key,
                    region=self._azure_region,
                    speech_recognition_language=self._language
                )
                audio_config = speechsdk.AudioConfig(filename=wav_path)
                recognizer   = speechsdk.SpeechRecognizer(
                    speech_config=speech_config,
                    audio_config=audio_config
                )
                r    = recognizer.recognize_once()
                text = r.text.strip() if r.text else ""
                print(f"[Speech {result.chunk_id}] '{text}'")
            except Exception as e:
                print(f"[Speech {result.chunk_id}] Error: {e}")
                text = ""
        result.t_azure_end = time.time()
        result.set_speech(text)

    # =========================================================================
    # Per-chunk: speaker recognition
    # =========================================================================

    def _speaker_thread(self, result: _ChunkResult, wav_path: str):
        result.t_speaker_start = time.time()
        try:
            if self.speaker_recognizer is None or not self.speaker_recognizer.profiles:
                result.set_speaker("unknown", 0.0)
                return
            speaker, score = self.speaker_recognizer.identify_from_wav(wav_path)
            print(f"[Speaker {result.chunk_id}] {speaker} (score={score:.2f})")
        except Exception as e:
            print(f"[Speaker {result.chunk_id}] Error: {e}")
            speaker, score = "unknown", 0.0
        result.t_speaker_end = time.time()
        result.set_speaker(speaker, score)

    # =========================================================================
    # Collector thread — assemble results in integer ID order
    # =========================================================================

    def _collector_thread(self):
        """
        Blocks on each _ChunkResult in the order they were recorded.
        When both speech and speaker sub-tasks finish, appends a clean entry
        to self.detected_speech (skipping empty or robot-labelled results).
        """
        while self.running:
            try:
                result = self._collect_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if not result.wait(timeout=15.0):
                print(f"[Collector] Chunk {result.chunk_id} timed out, skipping.")
                continue

            text    = result.text
            speaker = result.speaker
            score   = result.score

            if not text:
                continue
            if speaker == "robot":
                print(f"[Collector] Ignoring robot speech (chunk {result.chunk_id}).")
                continue
            result.t_collected = time.time()
            processing_latency   = result.t_collected   - result.t_flushed
            print(f"[Collector] chunk={result.chunk_id} "
                  f"speaker={speaker} score={score:.2f} text='{text}'"
                  f"azure_latency={(result.t_azure_end - result.t_azure_start):.2f}s"
                  f" processing_latency={processing_latency:.2f}s")
            self.detected_speech.append({"speaker": speaker, "text": text})
            
            self.t_last_chunk_collected = result.t_collected
            # Remove WAV to avoid filling disk
            wav_path = os.path.join(self._tmp_dir, f"chunk_{result.chunk_id:06d}.wav")
            try:
                os.remove(wav_path)
            except OSError:
                pass

    # =========================================================================
    # Turn assembly
    # =========================================================================

    def recognized_something(self) -> bool:
        """
        Called after FINAL_SILENCE_TIME expires.
        Assembles all collected chunks into self.recognized_text and
        self.final_speaker.  Clears self.detected_speech.
        Returns True when there is something meaningful to send.
        """
        if not self.detected_speech:
            return False

        full_text: str  = ""
        speakers: set   = set()

        for entry in self.detected_speech:
            full_text += entry["text"] + " "
            if entry["speaker"] != "unknown":
                speakers.add(entry["speaker"])

        self.recognized_text = full_text.strip()
        self.detected_speech = []

        if len(speakers) > 1:
            print("[Turn] Multiple speakers:", speakers)
            self.final_speaker = "multiple"
        elif len(speakers) == 0:
            self.final_speaker = "unknown"
        else:
            self.final_speaker = next(iter(speakers))

        print(f"[Turn] speaker={self.final_speaker} text='{self.recognized_text}'")
        return bool(self.recognized_text)

    # =========================================================================
    # Socket / XML helpers  (same protocol as before)
    # =========================================================================
    @staticmethod
    def _wrap_xml(text: str, user_id: str = "user1",
                  lang: str = "it-IT", speaking_time: int = 1) -> str:
        return (
            f'<response>\n'
            f'    <profile_id value="{user_id}">\n'
            f'        {text}\n'
            f'        <language>{lang}</language>\n'
            f'        <speaking_time>{speaking_time}</speaking_time>\n'
            f'    </profile_id>\n'
            f'</response>'
        )

    def _notify_emoact(self, text: str):
        """Fire-and-forget POST to EmoACT — runs in a background thread."""
        t0 = time.time()
        try:
            requests.post(
                self.url_emoACT + "/sentence_analysis",
                json={"sentence": text},
                timeout=2
            )
            print(f"[STT] Sent to EmoACT in {time.time() - t0:.2f}s")
        except Exception:
            print(f"[STT] EmoACT unreachable after {time.time() - t0:.2f}s, skipping.")

    def _send_recognized(self, conn: socket.socket):
        text = self.recognized_text
        print(f"[STT] Sending turn: {text}")

        # Notify EmoACT (best-effort)
        if self.emoACT_active:
            threading.Thread(target=self._notify_emoact, args=(text,), daemon=True).start()

        xml = self._wrap_xml(text, user_id=self.final_speaker)
        if not self.pause_mode:
            self.speaking = True          # stop listening until robot finishes
        conn.send(xml.encode("utf-8"))
        end_to_send = time.time() - self.t_last_chunk_collected
        print(f"[STT] Sent XML to DM in {end_to_send:.2f}s")

    # =========================================================================
    # Main listen loop  (socket mode)
    # =========================================================================

    def listen(self):
        """
        Blocks until a client connects, then runs the turn-taking loop:
          1. Discard any partial audio while robot is speaking.
          2. Wait for FINAL_SILENCE_TIME with no new speech chunks.
             Reset the window whenever a new chunk arrives.
          3. recognized_something() → send XML ; otherwise → send "timeout".
          4. Wait for client "ready" before the next turn.
        """
        print("[Socket] Waiting for client…")
        conn, addr = self.server.accept()
        print(f"[Socket] Client connected: {addr}")
        conn.recv(256).decode("utf-8")   # initial handshake / ready

        self.start()

        while True:
            # ---- wait while robot is speaking ----
            while self.speaking:
                print("[Listen] Robot speaking, discarding partial input…")
                self.detected_speech = []
                self.recognized_text = ""
                time.sleep(0.1)

            # ---- wait for FINAL_SILENCE_TIME with no new chunk ----
            print("[Listen] Listening for turn…")
            self.t_final      = time.time() + FINAL_SILENCE_TIME
            last_count = len(self.detected_speech)

            while time.time() < self.t_final:
                if self.speaking:
                    print("[Listen] Robot started speaking mid-turn.")
                    self.detected_speech = []
                    self.recognized_text = ""
                    break
                # New chunk arrived → extend the silence window
                current_count = len(self.detected_speech)
                if current_count > last_count:
                    last_count = current_count
                    t_end = time.time() + FINAL_SILENCE_TIME
                time.sleep(0.05)
            else:
                # Silence window expired naturally
                print("[Listen] ---- TIMEOUT ----")
                if self.speaking:
                    continue

                if self.recognized_something():
                    self._send_recognized(conn)
                else:
                    print("[Listen] Nothing recognized → sending timeout.")
                    conn.send("timeout".encode("utf-8"))

                # Wait for client ready signal
                print("[Listen] Waiting for client ready…")
                try:
                    msg = conn.recv(256).decode("utf-8")
                except Exception:
                    msg = ""
                if not msg:
                    print("[Socket] Client disconnected.")
                    break
                print("[Listen] Client ready, next turn.")
                self.speaking = False
                self.recognized_text = ""
                self.detected_speech = []

        self.server.close()

    # =========================================================================
    # Standalone listen loop  (no socket)
    # =========================================================================

    def simple_listen(self):
        """Standalone loop — prints recognized turns to stdout, no socket."""
        self.start()
        while True:
            while self.speaking:
                self.detected_speech = []
                self.recognized_text = ""
                time.sleep(0.1)

            t_end      = time.time() + FINAL_SILENCE_TIME
            last_count = len(self.detected_speech)

            while time.time() < t_end:
                if self.speaking:
                    self.detected_speech = []
                    break
                if len(self.detected_speech) > last_count:
                    last_count = len(self.detected_speech)
                    t_end = time.time() + FINAL_SILENCE_TIME
                time.sleep(0.05)
            else:
                if self.recognized_something():
                    print("[SimpleListen] Recognized:", self.recognized_text)
                    self.recognized_text = ""
                    self.detected_speech = []