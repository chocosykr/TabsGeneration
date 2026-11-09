#!/usr/bin/env python3
"""
TabsGeneration Interactive Web UI Server
Provides an audio player, STFT spectrogram visualization, interactive amplitude waveform visualizer,
target segment selection, and targeted single-chord & sequence transcription with Karplus-Strong acoustic synthesis.
"""

import os
import sys
import io
import json
import time
import base64
import urllib.parse
import mimetypes
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from typing import Optional, Dict, Any, List

# Ensure prototype directory is in Python path
PROTOTYPE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(PROTOTYPE_DIR)
if PROTOTYPE_DIR not in sys.path:
    sys.path.insert(0, PROTOTYPE_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import scipy.io.wavfile as wavfile
import librosa
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from device_config import DEVICE
from main import process_audio_multi_notes
from chord_sequence import identify_audio_chord_sequence
from fretboard_mapper import UKULELE_TUNING

# Storage for uploads and temporary synthesized outputs
UPLOAD_DIR = os.path.join(PROTOTYPE_DIR, "input", "uploads")
SYNTH_DIR = os.path.join(PROTOTYPE_DIR, "output_ui")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(SYNTH_DIR, exist_ok=True)


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Multi-threaded HTTP server for responsive concurrent playback and transcription."""
    daemon_threads = True


def get_available_audio_files() -> List[Dict[str, Any]]:
    """Scan the repository for WAV audio files available for testing."""
    files = []
    scan_dirs = [PROTOTYPE_DIR, os.path.join(PROTOTYPE_DIR, "input"), UPLOAD_DIR, PROJECT_ROOT]
    seen = set()

    for d in scan_dirs:
        if not os.path.exists(d):
            continue
        for entry in sorted(os.listdir(d)):
            if entry.lower().endswith((".wav", ".mp3", ".ogg", ".flac")):
                full_path = os.path.abspath(os.path.join(d, entry))
                if full_path in seen or not os.path.isfile(full_path):
                    continue
                seen.add(full_path)
                try:
                    size_kb = os.path.getsize(full_path) / 1024.0
                    rel_path = os.path.relpath(full_path, PROJECT_ROOT)
                    files.append({
                        "name": entry,
                        "path": full_path,
                        "rel_path": rel_path,
                        "size_kb": round(size_kb, 1)
                    })
                except Exception:
                    pass
    return files


def get_audio_waveform_data(audio_path: str, bars_per_sec: int = 50) -> Dict[str, Any]:
    """Extract amplitude envelope peaks, RMS energy, and physical onset attack timestamps."""
    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    dur = float(len(y) / sr)

    num_bars = max(40, int(dur * bars_per_sec))
    chunk_size = len(y) // num_bars
    if chunk_size > 0:
        chunks = y[:chunk_size * num_bars].reshape(num_bars, chunk_size)
        peaks = np.max(np.abs(chunks), axis=1)
        rms = np.sqrt(np.mean(chunks ** 2, axis=1))
    else:
        peaks = np.abs(y)
        rms = peaks.copy()

    max_peak = float(np.max(peaks)) if len(peaks) > 0 and np.max(peaks) > 0 else 1.0
    peaks_norm = np.clip(peaks / max_peak, 0.02, 1.0)

    max_rms = float(np.max(rms)) if len(rms) > 0 and np.max(rms) > 0 else 1.0
    rms_norm = np.clip(rms / max_rms, 0.01, 1.0)

    # Detect attack onsets (plucks & strums)
    try:
        onsets = librosa.onset.onset_detect(y=y, sr=sr, units='time').tolist()
    except Exception:
        onsets = []

    return {
        "duration": dur,
        "sample_rate": sr,
        "num_bars": num_bars,
        "peaks": [round(float(p), 4) for p in peaks_norm],
        "rms": [round(float(r), 4) for r in rms_norm],
        "onsets": [round(float(t), 3) for t in onsets]
    }


def generate_spectrogram_image(audio_path: str,
                               start_sec: Optional[float] = None,
                               end_sec: Optional[float] = None,
                               show_guides: bool = True) -> Dict[str, Any]:
    """Generate high-resolution STFT spectrogram with toggleable ukulele pitch guides."""
    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    total_duration = len(y) / sr

    n_fft = 1024
    hop_length = 256
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length)), ref=np.max)

    fig, ax = plt.subplots(figsize=(11, 4.2), dpi=120)
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#0f172a')

    librosa.display.specshow(D, sr=sr, hop_length=hop_length, x_axis='time', y_axis='linear',
                             ax=ax, cmap='magma')
    ax.set_ylim(0, 1600)

    ax.tick_params(colors='#94a3b8', labelsize=10)
    ax.xaxis.label.set_color('#94a3b8')
    ax.yaxis.label.set_color('#94a3b8')
    ax.set_xlabel("Time (seconds)", fontsize=11, fontweight='bold', labelpad=6)
    ax.set_ylabel("Frequency (Hz)", fontsize=11, fontweight='bold', labelpad=6)
    for spine in ax.spines.values():
        spine.set_color('#334155')

    if show_guides:
        ukulele_strings = [
            ("C4 (String 3, 261.6 Hz)", UKULELE_TUNING[3]["freq"], "#38bdf8"),
            ("E4 (String 2, 329.6 Hz)", UKULELE_TUNING[2]["freq"], "#a855f7"),
            ("G4 (String 4, 392.0 Hz)", UKULELE_TUNING[4]["freq"], "#4ade80"),
            ("A4 (String 1, 440.0 Hz)", UKULELE_TUNING[1]["freq"], "#facc15"),
        ]
        for label, freq, color in ukulele_strings:
            ax.axhline(freq, color=color, linestyle="--", linewidth=1.0, alpha=0.75)
            ax.text(total_duration * 0.015, freq + 15, label, color=color, fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#0f172a", edgecolor=color, alpha=0.8))

    if start_sec is not None and end_sec is not None and end_sec > start_sec:
        s = max(0.0, float(start_sec))
        e = min(total_duration, float(end_sec))
        ax.axvspan(s, e, color="#38bdf8", alpha=0.25)
        ax.axvline(s, color="#0ea5e9", linewidth=2.0, linestyle="-")
        ax.axvline(e, color="#0ea5e9", linewidth=2.0, linestyle="-")

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")

    return {
        "image_base64": img_b64,
        "duration": float(total_duration),
        "sample_rate": sr,
        "samples": len(y)
    }


class UIRequestHandler(BaseHTTPRequestHandler):
    """Custom HTTP Handler serving Frontend UI and REST APIs with audio streaming."""

    def log_message(self, format, *args):
        sys.stderr.write(f"[{time.strftime('%H:%M:%S')}] {args[0]} {args[1]}\n")

    def end_cors_headers(self, status=200, content_type="application/json"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS, HEAD")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Range")
        self.send_header("Accept-Ranges", "bytes")

    def do_OPTIONS(self):
        self.end_cors_headers(204)
        self.end_headers()

    def do_HEAD(self):
        self.do_GET(head_only=True)

    def do_GET(self, head_only: bool = False):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        if path == "/" or path == "/index.html":
            self.serve_frontend(head_only=head_only)
            return

        if path == "/api/files":
            files = get_available_audio_files()
            resp = json.dumps({"files": files}).encode("utf-8")
            self.end_cors_headers(200, "application/json")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            if not head_only:
                self.wfile.write(resp)
            return

        if path == "/api/audio":
            file_param = query.get("file", [None])[0]
            if not file_param or not os.path.exists(file_param):
                self.send_error(404, "Audio file not found")
                return
            self.stream_audio_file(file_param, head_only=head_only)
            return

        if path == "/api/waveform":
            file_param = query.get("file", [None])[0]
            if not file_param or not os.path.exists(file_param):
                self.send_json({"error": "File not found"}, 404)
                return
            try:
                wave_data = get_audio_waveform_data(file_param)
                self.send_json(wave_data)
            except Exception as e:
                self.send_json({"error": str(e)}, 500)
            return

        self.send_error(404, "Not Found")

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/api/spectrogram":
            data = self.read_json_body()
            audio_path = data.get("file")
            if not audio_path or not os.path.exists(audio_path):
                self.send_json({"error": f"Audio file not found: {audio_path}"}, 400)
                return
            show_guides = bool(data.get("show_guides", True))
            try:
                spec_data = generate_spectrogram_image(
                    audio_path, data.get("start_sec"), data.get("end_sec"),
                    show_guides=show_guides)
                self.send_json(spec_data)
            except Exception as e:
                self.send_json({"error": str(e)}, 500)
            return

        if path == "/api/transcribe_segment":
            data = self.read_json_body()
            audio_path = data.get("file")
            if not audio_path or not os.path.exists(audio_path):
                self.send_json({"error": f"Audio file not found: {audio_path}"}, 400)
                return
            strummed = [int(s.strip()) for s in data.get("strummed", "1,2,3,4").split(",") if s.strip().isdigit()]
            synth_out_path = os.path.join(SYNTH_DIR, f"synth_single_{int(time.time()*1000)}.wav")
            try:
                res = process_audio_multi_notes(
                    file_path=audio_path,
                    num_notes=int(data.get("notes", 4)),
                    strummed_strings=strummed,
                    num_iterations=int(data.get("iter", 25)),
                    lr=float(data.get("lr", 0.5)),
                    top_n=int(data.get("top_n", 6)),
                    save_synth=True,
                    synth_out_path=synth_out_path,
                    start_sec=float(data.get("start_sec", 0.0)),
                    end_sec=float(data.get("end_sec", 0.5))
                )
                self.send_json(self.format_single_chord_result(res, synth_out_path))
            except Exception as e:
                import traceback; traceback.print_exc()
                self.send_json({"error": str(e)}, 500)
            return

        if path == "/api/transcribe_sequence":
            data = self.read_json_body()
            audio_path = data.get("file")
            if not audio_path or not os.path.exists(audio_path):
                self.send_json({"error": f"Audio file not found: {audio_path}"}, 400)
                return
            synth_out_path = os.path.join(SYNTH_DIR, f"synth_seq_{int(time.time()*1000)}.wav")
            try:
                segments = identify_audio_chord_sequence(
                    audio_path=audio_path,
                    min_duration=data.get("min_duration", "auto"),
                    split_strums=bool(data.get("split_strums", False)),
                    save_synth=True,
                    synth_out_path=synth_out_path,
                    start_sec=float(data["start_sec"]) if data.get("start_sec") is not None else None,
                    end_sec=float(data["end_sec"]) if data.get("end_sec") is not None else None
                )
                self.send_json(self.format_sequence_result(segments, synth_out_path))
            except Exception as e:
                import traceback; traceback.print_exc()
                self.send_json({"error": str(e)}, 500)
            return

        if path == "/api/upload":
            self.handle_file_upload()
            return

        self.send_error(404, "Not Found")

    def read_json_body(self) -> Dict[str, Any]:
        content_len = int(self.headers.get("Content-Length", 0))
        if content_len == 0:
            return {}
        try:
            return json.loads(self.rfile.read(content_len).decode("utf-8"))
        except Exception:
            return {}

    def send_json(self, payload: Dict[str, Any], status: int = 200):
        resp = json.dumps(payload).encode("utf-8")
        self.end_cors_headers(status, "application/json")
        self.send_header("Content-Length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)

    def handle_file_upload(self):
        content_type = self.headers.get("Content-Type", "")
        content_len = int(self.headers.get("Content-Length", 0))
        if content_len == 0:
            self.send_json({"error": "Empty upload"}, 400)
            return
        body = self.rfile.read(content_len)
        filename = f"upload_{int(time.time())}.wav"
        if "boundary=" in content_type:
            boundary = content_type.split("boundary=")[1].encode()
            for p in body.split(b"--" + boundary):
                if b'filename="' in p:
                    hdr, content_part = p.split(b"\r\n\r\n", 1)
                    fname = hdr.split(b'filename="')[1].split(b'"')[0].decode(errors="ignore")
                    if fname:
                        filename = fname
                    body = content_part.rstrip(b"\r\n--")
                    break
        dest_path = os.path.join(UPLOAD_DIR, filename)
        with open(dest_path, "wb") as f:
            f.write(body)
        self.send_json({"success": True, "filename": filename, "path": dest_path})

    def stream_audio_file(self, file_path: str, head_only: bool = False):
        file_size = os.path.getsize(file_path)
        mime_type, _ = mimetypes.guess_type(file_path)
        if not mime_type:
            mime_type = "audio/wav"
        range_header = self.headers.get("Range", None)
        if range_header:
            try:
                parts = range_header.strip().split("=")[1].split("-")
                start = int(parts[0]) if parts[0] else 0
                end = int(parts[1]) if len(parts) > 1 and parts[1] else file_size - 1
            except Exception:
                start, end = 0, file_size - 1
            start = max(0, min(start, file_size - 1))
            end = max(start, min(end, file_size - 1))
            length = end - start + 1
            self.send_response(206)
            self.send_header("Content-Type", mime_type)
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
            self.send_header("Content-Length", str(length))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            if head_only:
                return
            with open(file_path, "rb") as f:
                f.seek(start)
                remaining = length
                while remaining > 0:
                    buf = f.read(min(65536, remaining))
                    if not buf:
                        break
                    self.wfile.write(buf)
                    remaining -= len(buf)
        else:
            self.send_response(200)
            self.send_header("Content-Type", mime_type)
            self.send_header("Content-Length", str(file_size))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            if head_only:
                return
            with open(file_path, "rb") as f:
                while True:
                    buf = f.read(65536)
                    if not buf:
                        break
                    self.wfile.write(buf)

    def format_single_chord_result(self, res, synth_path):
        if res.get("type") == "single_note":
            return {
                "mode": "single_note",
                "notes": [res.get("note_name", "")],
                "tab_4321": [0, 0, 0, int(res.get("fret", 0))],
                "acoustic_match": 100.0, "stft_loss": 0.0,
                "elapsed": round(res.get("elapsed", 0.0), 2),
                "combined_score": 100.0,
                "synth_url": f"/api/audio?file={urllib.parse.quote(synth_path)}" if os.path.exists(synth_path) else None,
                "candidates": [{"rank": i+1, "notes": c.get("open_name",""), "tab": f"String {c['string']} Fret {c['fret']}",
                                "tab_4321": [0,0,0,c['fret']], "audio_match": round(c.get("audio_match_norm",1)*100,1),
                                "spectral_fit": round(c.get("inharm_score",0)*100,1), "combined": 100.0,
                                "verdict": "WINNER" if i==0 else "Rejected"}
                               for i, c in enumerate(res.get("candidates", []))]
            }
        winner = res.get("winner", {})
        mapping = res.get("mapping", {})
        note_names = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
        cands = []
        for rank, c in enumerate(res.get("candidates", []), 1):
            midis = c.get("multiset_midis", [])
            notes = [f"{note_names[m%12]}{m//12-1}" for m in midis]
            c_map = c.get("refined", {}).get("mapping", c.get("mapping", {}))
            cands.append({
                "rank": rank, "notes": ", ".join(notes),
                "tab": " - ".join(str(c_map.get(s, 0)) for s in [4,3,2,1]),
                "tab_4321": [c_map.get(s, 0) for s in [4,3,2,1]],
                "audio_match": round(c.get("audio_score_norm", 1.0)*100, 1),
                "spectral_fit": round(c.get("spectral_score_norm", 0.0)*100, 1),
                "combined": round(c.get("combined_score", 0.0)*100, 1),
                "stft_loss": round(float(c.get("audio_loss", 0.0)), 4),
                "verdict": "WINNER" if rank == 1 else "Rejected"
            })
        return {
            "mode": "chord",
            "notes": res.get("notes", []),
            "tab_4321": res.get("tab_4321", [0,0,0,0]),
            "acoustic_match": round(res.get("audio_match", 100.0), 1),
            "stft_loss": round(res.get("stft_loss", 0.0), 4),
            "combined_score": round(res.get("combined_score", 100.0), 1),
            "elapsed": round(res.get("elapsed", 0.0), 2),
            "device": torch.cuda.get_device_name(DEVICE) if DEVICE.type == "cuda" else "CPU",
            "synth_url": f"/api/audio?file={urllib.parse.quote(synth_path)}" if os.path.exists(synth_path) else None,
            "candidates": cands
        }

    def format_sequence_result(self, segments, synth_path):
        seg_list = []
        for idx, s in enumerate(segments, 1):
            c = s.get("chord_data", {})
            seg_list.append({
                "index": idx, "chord": s.get("chord", "?"),
                "quality": c.get("quality", "Major"),
                "start": round(s.get("start", 0.0), 2), "end": round(s.get("end", 0.0), 2),
                "duration": round(s.get("duration", 0.0), 2),
                "strum_count": s.get("strum_count", 1),
                "tab_4321": c.get("tab_4321", [0,0,0,0]),
                "notes": c.get("notes_4321", []),
                "confidence": round(s.get("confidence", 0.0)*100.0, 1),
                "stft_loss": round(float(s.get("stft_loss", 0.0)), 4)
            })
        prog_str = " -> ".join(f"[{s['chord']} {s['quality']}]" for s in seg_list)
        return {
            "mode": "sequence", "progression_flow": prog_str,
            "total_chords": len(segments), "segments": seg_list,
            "synth_url": f"/api/audio?file={urllib.parse.quote(synth_path)}" if os.path.exists(synth_path) else None
        }

    def serve_frontend(self, head_only: bool = False):
        html = get_frontend_html()
        content = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        if not head_only:
            self.wfile.write(content)


def get_frontend_html() -> str:
    return r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>TabsGeneration – Ukulele Audio Inspector</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    body { background:#0b0f19; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }
    input[type=range]{accent-color:#38bdf8;}
    ::-webkit-scrollbar{width:8px;height:8px;}
    ::-webkit-scrollbar-track{background:#0f172a;}
    ::-webkit-scrollbar-thumb{background:#334155;border-radius:4px;}
    ::-webkit-scrollbar-thumb:hover{background:#475569;}
    #waveformScrollContainer{cursor:crosshair;user-select:none;}
    .fretboard-dot{transition:all .2s;}
  </style>
</head>
<body class="text-slate-100 min-h-screen p-4 md:p-6">
<div class="max-w-7xl mx-auto space-y-5">

  <!-- ── HEADER ── -->
  <header class="flex flex-col md:flex-row md:items-center justify-between pb-4 border-b border-slate-800 gap-3">
    <div>
      <div class="flex items-center gap-2.5">
        <span class="text-2xl">🎸</span>
        <h1 class="text-xl font-bold bg-gradient-to-r from-sky-400 via-indigo-300 to-purple-400 bg-clip-text text-transparent">
          TabsGeneration – Audio & Tab Inspector
        </h1>
        <span class="px-2 py-0.5 text-[11px] font-semibold rounded bg-sky-950 text-sky-400 border border-sky-800/60">Targeted Segment Execution</span>
      </div>
      <p class="text-xs text-slate-400 mt-1">Select a segment on the waveform, inspect the spectrogram, then run transcription.</p>
    </div>
    <div class="flex items-center gap-2">
      <label class="cursor-pointer px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 text-xs font-medium rounded-lg flex items-center gap-1.5">
        📁 Upload <input type="file" id="fileUploadInput" accept="audio/*" class="hidden">
      </label>
      <span class="px-2.5 py-1.5 text-xs bg-slate-800/80 text-emerald-400 border border-slate-700 rounded-lg">Device: CPU</span>
    </div>
  </header>

  <!-- ── MAIN GRID ── -->
  <div class="grid grid-cols-1 lg:grid-cols-3 gap-5">

    <!-- LEFT PANEL -->
    <div class="bg-slate-900 border border-slate-800 rounded-xl p-4 shadow-lg space-y-4">

      <!-- File Select -->
      <div class="flex items-center justify-between">
        <h2 class="text-sm font-semibold text-slate-200">1. Select Recording</h2>
        <span id="audioDurBadge" class="text-xs font-mono text-sky-400 bg-sky-950/50 px-2 py-0.5 rounded border border-sky-800/50">0.00s</span>
      </div>
      <select id="audioSelect" class="w-full bg-slate-800 border border-slate-700 text-slate-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-sky-500">
        <option value="">Loading…</option>
      </select>

      <!-- ══ WAVEFORM VISUALIZER ══ -->
      <div class="bg-[#0a0f1c] border border-slate-800 rounded-xl p-3 space-y-2.5">
        <!-- toolbar -->
        <div class="flex items-center justify-between gap-2">
          <span class="text-xs font-semibold text-slate-200">📊 Amplitude Timeline</span>
          <div class="flex items-center gap-1">
            <button id="zoomFitBtn"  class="wave-zoom-btn bg-slate-800 text-slate-300" title="Fit to width">Fit</button>
            <button id="zoom1xBtn"   class="wave-zoom-btn bg-sky-600 text-white"       title="1× scroll (80 px/s)">1×</button>
            <button id="zoom2xBtn"   class="wave-zoom-btn bg-slate-800 text-slate-300" title="2× scroll (160 px/s)">2×</button>
            <button id="zoom4xBtn"   class="wave-zoom-btn bg-slate-800 text-slate-300" title="4× scroll (320 px/s)">4×</button>
            <button id="toggleMetricBtn" class="wave-zoom-btn bg-slate-800 text-slate-400 ml-1" title="Peak / RMS">Peak</button>
          </div>
        </div>
        <style>.wave-zoom-btn{padding:1px 7px;font-size:10px;font-weight:600;border-radius:5px;cursor:pointer;transition:all .15s;}</style>

        <!-- Scrollable canvas wrapper with centered completion line -->
        <div class="relative w-full overflow-hidden rounded-lg bg-black/80 border border-slate-800">
          <!-- Center completion needle (stays at horizontal center during playback/scrub) -->
          <div id="centerNeedleGuide" class="pointer-events-none absolute top-0 bottom-0 left-1/2 w-0.5 -ml-[1px] bg-sky-400/90 z-20 shadow-[0_0_8px_rgba(56,189,248,0.9)] flex flex-col items-center">
            <div class="w-2.5 h-2.5 -mt-0.5 rotate-45 bg-sky-400 border border-white"></div>
            <div class="flex-1 w-0.5 bg-sky-400/80"></div>
            <div class="w-1.5 h-1.5 -mb-0.5 rounded-full bg-white"></div>
          </div>

          <div id="waveformScrollContainer" class="w-full overflow-x-auto overflow-y-hidden">
            <canvas id="waveformCanvas" height="120" class="block"></canvas>
          </div>
        </div>

        <!-- Legend row -->
        <div class="flex items-center justify-between text-[11px] text-slate-400 px-0.5">
          <span class="flex items-center gap-1.5">
            <span class="inline-block w-2 h-2 rounded-full bg-[#38bdf8]"></span>Played
            <span class="inline-block w-2 h-2 rounded-full bg-[#f59e0b] ml-2"></span>Strum onsets
          </span>
          <button id="centerPlayheadToggle" class="text-sky-400 hover:underline text-[11px] font-semibold">Center Playhead: ON</button>
        </div>

        <!-- Strum Onset Quick-select chips -->
        <div id="onsetsContainer" class="flex flex-wrap items-center gap-1 text-[11px] text-slate-400 pt-1 border-t border-slate-800/80">
          <span class="text-slate-500 font-mono text-[10px]">Onsets:</span>
          <span id="onsetsList" class="text-slate-500 italic text-[10px]">None detected</span>
        </div>
      </div>
      <!-- ══ END WAVEFORM ══ -->

      <!-- Audio Player -->
      <div class="bg-slate-950/70 border border-slate-800/80 rounded-lg p-3 space-y-2">
        <div class="flex justify-between items-center">
          <span class="text-xs font-semibold text-slate-300">Audio Player</span>
          <span id="currentTimeDisplay" class="text-xs font-mono text-sky-400">0.00 / 0.00s</span>
        </div>
        <audio id="mainAudioPlayer" class="w-full" controls preload="metadata"></audio>
        <div class="flex gap-2">
          <button id="playSegmentBtn" class="flex-1 py-1.5 bg-sky-600 hover:bg-sky-500 text-white rounded-lg text-xs font-semibold transition">▶ Play Segment</button>
          <button id="loopSegmentBtn" class="px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 rounded-lg text-xs transition">🔁 Loop</button>
        </div>
      </div>

      <!-- Segment Controls -->
      <div class="space-y-2.5">
        <div class="flex items-center justify-between">
          <label class="text-xs font-semibold text-slate-300">2. Target Segment</label>
          <button id="selectAllBtn" class="text-xs text-sky-400 hover:underline">Select All</button>
        </div>
        <div class="grid grid-cols-2 gap-2.5">
          <div>
            <label class="block text-[11px] text-slate-400 mb-1">Start (sec)</label>
            <input type="number" id="startTimeInput" step="0.05" min="0" value="0.00"
                   class="w-full bg-slate-800 border border-slate-700 text-slate-200 rounded px-2 py-1.5 text-xs font-mono">
          </div>
          <div>
            <label class="block text-[11px] text-slate-400 mb-1">End (sec)</label>
            <input type="number" id="endTimeInput" step="0.05" min="0" value="0.60"
                   class="w-full bg-slate-800 border border-slate-700 text-slate-200 rounded px-2 py-1.5 text-xs font-mono">
          </div>
        </div>
        <div class="space-y-1.5">
          <div class="flex justify-between text-[10px] text-slate-500"><span>Start</span><span id="startSliderLabel">0.00s</span></div>
          <input type="range" id="startSlider" min="0" max="10" step="0.01" value="0" class="w-full h-1 bg-slate-700 rounded-lg cursor-pointer">
          <div class="flex justify-between text-[10px] text-slate-500"><span>End</span><span id="endSliderLabel">0.60s</span></div>
          <input type="range" id="endSlider" min="0" max="10" step="0.01" value="0.6" class="w-full h-1 bg-slate-700 rounded-lg cursor-pointer">
        </div>
        <div class="text-xs text-slate-400 bg-slate-950/40 px-2 py-1.5 rounded border border-slate-800/60 flex justify-between">
          <span>Duration:</span>
          <span id="segmentDurationLabel" class="font-mono text-sky-400 font-semibold">0.60s</span>
        </div>
      </div>

      <!-- Action Buttons -->
      <div class="space-y-2 pt-2 border-t border-slate-800">
        <button id="runChordBtn" class="w-full py-2.5 bg-gradient-to-r from-sky-500 to-indigo-600 hover:from-sky-400 hover:to-indigo-500 text-white font-semibold rounded-lg text-sm transition shadow flex items-center justify-center gap-1.5">
          ⚡ Transcribe Segment (Single Chord)
        </button>
        <button id="runSequenceBtn" class="w-full py-2 bg-slate-800 hover:bg-slate-700 text-indigo-300 hover:text-white border border-slate-700 font-medium rounded-lg text-xs transition flex items-center justify-center gap-1.5">
          🎼 Transcribe Progression (Sequence)
        </button>
      </div>
    </div>

    <!-- RIGHT PANEL – Spectrogram -->
    <div class="lg:col-span-2 bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-lg flex flex-col">
      <div class="flex items-center justify-between pb-3 border-b border-slate-800 gap-3 flex-wrap">
        <div>
          <h2 class="text-sm font-semibold text-slate-200">Spectrogram · STFT Harmonic View</h2>
          <p class="text-[11px] text-slate-400 mt-0.5">Blue selection box = targeted segment. Toggle string guides on/off below.</p>
        </div>
        <div class="flex items-center gap-2">
          <button id="toggleGuidesBtn"
            class="px-3 py-1.5 text-xs font-semibold rounded-lg border border-sky-600/80 bg-sky-950/80 text-sky-300 hover:bg-sky-900/60 transition flex items-center gap-1.5">
            🎸 String Guides: ON
          </button>
          <span id="specStatus" class="text-xs text-slate-500">Ready</span>
        </div>
      </div>

      <div class="relative bg-slate-950 rounded-lg border border-slate-800/90 overflow-hidden my-3 min-h-[260px] flex items-center justify-center flex-1">
        <img id="spectrogramImg" class="w-full h-auto object-contain hidden" alt="Spectrogram">
        <div id="specLoading" class="text-center py-16 text-slate-500 text-sm">Select an audio file to view its spectrogram.</div>
      </div>

      <div id="stringLegendContainer" class="flex flex-wrap items-center justify-between gap-2 text-xs text-slate-400 pt-2 border-t border-slate-800/80">
        <div class="flex items-center gap-3">
          <span class="flex items-center gap-1.5"><span class="w-2.5 h-2.5 rounded-full bg-[#38bdf8]"></span>C4 (261 Hz)</span>
          <span class="flex items-center gap-1.5"><span class="w-2.5 h-2.5 rounded-full bg-[#a855f7]"></span>E4 (330 Hz)</span>
          <span class="flex items-center gap-1.5"><span class="w-2.5 h-2.5 rounded-full bg-[#4ade80]"></span>G4 (392 Hz)</span>
          <span class="flex items-center gap-1.5"><span class="w-2.5 h-2.5 rounded-full bg-[#facc15]"></span>A4 (440 Hz)</span>
        </div>
        <span class="text-slate-500">Blue box = selected segment</span>
      </div>
    </div>
  </div>

  <!-- ── RESULTS ── -->
  <div id="resultsSection" class="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-xl space-y-5 hidden">
    <div class="flex flex-col md:flex-row md:items-center justify-between pb-4 border-b border-slate-800 gap-3">
      <div>
        <h2 class="text-lg font-bold text-slate-100 flex items-center gap-2">
          🎯 Transcription Results
          <span id="resultModeBadge" class="text-xs px-2.5 py-0.5 rounded-full bg-indigo-950 text-indigo-400 border border-indigo-800">Analysis-by-Synthesis</span>
        </h2>
        <p id="resultSubtitle" class="text-xs text-slate-400 mt-0.5">Targeted evaluation on selected audio slice.</p>
      </div>
      <div class="flex items-center gap-4">
        <div class="text-right">
          <span class="block text-[11px] text-slate-400">Audio Match</span>
          <span id="acousticMatchBadge" class="text-lg font-bold text-emerald-400">--%</span>
        </div>
        <div class="text-right">
          <span class="block text-[11px] text-slate-400">STFT Loss</span>
          <span id="stftLossBadge" class="text-lg font-mono text-sky-400">--</span>
        </div>
      </div>
    </div>

    <!-- Single Chord -->
    <div id="singleChordContainer" class="grid grid-cols-1 md:grid-cols-12 gap-5">
      <div class="md:col-span-5 bg-slate-950/80 border border-slate-800 rounded-xl p-4 flex flex-col items-center">
        <h3 class="text-xs font-semibold text-slate-300 mb-2 w-full text-center uppercase tracking-wide">Fretboard Fingering</h3>
        <div id="fretboardSvgContainer" class="w-full flex justify-center"></div>
        <div class="mt-2 text-xs text-slate-400">Tab (G·C·E·A):
          <span id="tabStringDisplay" class="font-mono font-bold text-sky-400 ml-1">--</span>
        </div>
      </div>
      <div class="md:col-span-7 space-y-4">
        <div class="grid grid-cols-3 gap-2">
          <div class="bg-slate-950/80 border border-slate-800/80 p-3 rounded-lg text-center">
            <span class="block text-[11px] text-slate-400">Notes</span>
            <span id="chordNotesDisplay" class="text-sm font-bold text-indigo-300 font-mono mt-1 block">--</span>
          </div>
          <div class="bg-slate-950/80 border border-slate-800/80 p-3 rounded-lg text-center">
            <span class="block text-[11px] text-slate-400">Exec Time</span>
            <span id="elapsedDisplay" class="text-sm font-bold text-slate-200 mt-1 block">--</span>
          </div>
          <div class="bg-slate-950/80 border border-slate-800/80 p-3 rounded-lg text-center">
            <span class="block text-[11px] text-slate-400">Score</span>
            <span id="combinedScoreDisplay" class="text-sm font-bold text-emerald-400 mt-1 block">--</span>
          </div>
        </div>
        <div class="bg-indigo-950/30 border border-indigo-900/40 rounded-xl p-3.5 space-y-1.5">
          <div class="flex justify-between text-xs"><span class="font-semibold text-indigo-300">🔊 Synthesized Tab Audio (Karplus-Strong)</span><span class="text-slate-400 text-[11px]">Compare by ear</span></div>
          <audio id="synthAudioPlayer" class="w-full" controls></audio>
        </div>
        <div>
          <h4 class="text-[11px] font-semibold text-slate-300 uppercase tracking-wider mb-1.5">Candidate Rankings</h4>
          <div class="overflow-x-auto max-h-44 border border-slate-800 rounded-lg">
            <table class="w-full text-xs text-left text-slate-300">
              <thead class="bg-slate-950 text-slate-400 uppercase text-[10px]">
                <tr><th class="py-1.5 px-3">#</th><th class="py-1.5 px-3">Notes</th><th class="py-1.5 px-3">Tab</th><th class="py-1.5 px-3">Audio</th><th class="py-1.5 px-3">Spectral</th><th class="py-1.5 px-3">Status</th></tr>
              </thead>
              <tbody id="candidatesTableBody" class="divide-y divide-slate-800/60 bg-slate-900/40"></tbody>
            </table>
          </div>
        </div>
      </div>
    </div>

    <!-- Sequence Results -->
    <div id="sequenceContainer" class="space-y-4 hidden">
      <div class="bg-slate-950/80 border border-slate-800 rounded-xl p-3.5">
        <span class="text-[11px] text-slate-400 block mb-1">Progression:</span>
        <div id="progressionFlowDisplay" class="text-sm font-semibold text-sky-400 font-mono"></div>
      </div>
      <div class="bg-indigo-950/30 border border-indigo-900/40 rounded-xl p-3.5 space-y-1.5">
        <div class="flex justify-between text-xs"><span class="font-semibold text-indigo-300">Full Sequence Audio</span></div>
        <audio id="seqSynthAudioPlayer" class="w-full" controls></audio>
      </div>
      <div class="overflow-x-auto border border-slate-800 rounded-xl">
        <table class="w-full text-xs text-left text-slate-300">
          <thead class="bg-slate-950 text-slate-400 uppercase text-[10px]">
            <tr><th class="py-2 px-3">#</th><th class="py-2 px-3">Time</th><th class="py-2 px-3">Dur</th><th class="py-2 px-3">Chord</th><th class="py-2 px-3">Tab (4-3-2-1)</th><th class="py-2 px-3">Notes</th><th class="py-2 px-3">Strums</th><th class="py-2 px-3">Match%</th></tr>
          </thead>
          <tbody id="sequenceTableBody" class="divide-y divide-slate-800/60 bg-slate-900/40"></tbody>
        </table>
      </div>
    </div>
  </div>

</div>
<!-- ══════════════ SCRIPTS ══════════════ -->
<script>
/* ── State ── */
let curFile = "", curDur = 0, waveData = null;
let metricMode = 'peaks';   // 'peaks' | 'rms'
let zoomMode   = '1x';      // 'fit' | '1x' | '2x' | '4x'
let showGuides = true;
let isLooping  = false;
let autoCenterPlayhead = true;
let isDragging = false, dragStartSec = 0;
let animFrameId = null;

/* ── DOM Refs ── */
const Q = id => document.getElementById(id);
const audioSel       = Q("audioSelect"),
      mainAudio      = Q("mainAudioPlayer"),
      synthAudio     = Q("synthAudioPlayer"),
      seqSynthAudio  = Q("seqSynthAudioPlayer"),
      specImg        = Q("spectrogramImg"),
      specLoading    = Q("specLoading"),
      specStatus     = Q("specStatus"),
      audioDurBadge  = Q("audioDurBadge"),
      curTimeDis     = Q("currentTimeDisplay"),
      startInp       = Q("startTimeInput"),
      endInp         = Q("endTimeInput"),
      startSlider    = Q("startSlider"),
      endSlider      = Q("endSlider"),
      startLbl       = Q("startSliderLabel"),
      endLbl         = Q("endSliderLabel"),
      durLbl         = Q("segmentDurationLabel"),
      waveCanvas     = Q("waveformCanvas"),
      waveScroll     = Q("waveformScrollContainer"),
      centerGuide    = Q("centerNeedleGuide"),
      onsetsList     = Q("onsetsList"),
      toggleGuides   = Q("toggleGuidesBtn"),
      legend         = Q("stringLegendContainer"),
      resultsSection = Q("resultsSection"),
      singleCont     = Q("singleChordContainer"),
      seqCont        = Q("sequenceContainer");

/* ── Init ── */
async function init() {
  const r = await fetch("/api/files");
  const { files } = await r.json();
  audioSel.innerHTML = "";
  files.forEach(f => {
    const o = document.createElement("option");
    o.value = f.path; o.textContent = `${f.name} (${f.size_kb} KB)`;
    audioSel.appendChild(o);
  });
  if (files.length) loadAudio(files[0].path);
}

/* ── Load audio + spectrogram + waveform ── */
async function loadAudio(path) {
  if (!path) return;
  curFile = path;
  mainAudio.src = `/api/audio?file=${encodeURIComponent(path)}`;
  specLoading.textContent = "Computing spectrogram & waveform…";
  specLoading.classList.remove("hidden");
  specImg.classList.add("hidden");
  specStatus.textContent = "Rendering…";

  // Fetch waveform data and spectrogram in parallel
  Promise.all([
    fetchWaveform(path),
    fetchSpectrogram(path, null, null)
  ]);
}

async function fetchWaveform(path) {
  try {
    const r = await fetch(`/api/waveform?file=${encodeURIComponent(path)}`);
    const d = await r.json();
    if (d.peaks) {
      waveData = d;
      curDur = d.duration;
      audioDurBadge.textContent = `${curDur.toFixed(2)}s`;
      startSlider.max = curDur; endSlider.max = curDur;
      startSlider.value = 0;
      endSlider.value = Math.min(0.65, curDur);

      // Default zoom: fit for short clips (<5s), 1x scrollable for full songs
      zoomMode = curDur > 5.0 ? '1x' : 'fit';
      updateZoomButtons();

      renderOnsetChips(d.onsets || []);
      syncFromSliders(false);
      drawWave();
      setTimeout(drawWave, 50); // safety redraw once container dimensions settle
    }
  } catch (e) { console.error("waveform fetch error", e); }
}

function renderOnsetChips(onsets) {
  if (!onsetsList) return;
  if (!onsets || onsets.length === 0) {
    onsetsList.innerHTML = `<span class="text-slate-500 italic text-[10px]">None detected</span>`;
    return;
  }
  onsetsList.innerHTML = "";
  onsets.slice(0, 12).forEach((t, idx) => {
    const btn = document.createElement("button");
    btn.className = "px-1.5 py-0.5 rounded bg-amber-950/70 hover:bg-amber-800 text-amber-300 border border-amber-800/80 font-mono text-[10px] transition";
    btn.textContent = `⚡ #${idx+1} ${t.toFixed(2)}s`;
    btn.title = `Snap segment to strum at ${t.toFixed(2)}s`;
    btn.addEventListener("click", () => {
      startInp.value = t.toFixed(2);
      endInp.value = Math.min(curDur, t + 0.6).toFixed(2);
      mainAudio.currentTime = t;
      syncFromInputs();
    });
    onsetsList.appendChild(btn);
  });
}

async function fetchSpectrogram(path, s, e) {
  try {
    const r = await fetch("/api/spectrogram", {
      method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({ file: path, start_sec: s, end_sec: e, show_guides: showGuides })
    });
    const d = await r.json();
    if (d.image_base64) {
      specImg.src = `data:image/png;base64,${d.image_base64}`;
      specImg.classList.remove("hidden");
      specLoading.classList.add("hidden");
      specStatus.textContent = "Ready";
      if (!curDur && d.duration) {
        curDur = d.duration;
        audioDurBadge.textContent = `${curDur.toFixed(2)}s`;
        startSlider.max = curDur; endSlider.max = curDur;
        endSlider.value = Math.min(0.65, curDur);
        syncFromSliders(false);
      }
    }
  } catch (e) { specLoading.textContent = "Error loading spectrogram"; specStatus.textContent = "Error"; }
}

/* ── Geometry & Layout Helpers ── */
function getWaveGeometry() {
  const containerW = Math.max(300, (waveScroll ? waveScroll.clientWidth : 300));
  let audioW = containerW;
  if (zoomMode === '1x') audioW = Math.max(containerW, curDur * 80);
  else if (zoomMode === '2x') audioW = Math.max(containerW, curDur * 160);
  else if (zoomMode === '4x') audioW = Math.max(containerW, curDur * 320);

  const padX = autoCenterPlayhead ? Math.round(containerW / 2) : 0;
  const totalW = audioW + 2 * padX;
  return { containerW, audioW, padX, totalW };
}

/* ── Waveform Canvas Renderer ── */
function drawWave() {
  if (!waveCanvas) return;
  const dpr = window.devicePixelRatio || 1;
  const H = 120;
  const { containerW, audioW, padX, totalW } = getWaveGeometry();

  waveCanvas.style.width  = `${totalW}px`;
  waveCanvas.style.height = `${H}px`;
  waveCanvas.width  = Math.round(totalW * dpr);
  waveCanvas.height = Math.round(H * dpr);

  // Center needle overlay visibility
  if (centerGuide) {
    if (autoCenterPlayhead) centerGuide.classList.remove("hidden");
    else centerGuide.classList.add("hidden");
  }

  const ctx = waveCanvas.getContext("2d");
  ctx.clearRect(0, 0, waveCanvas.width, waveCanvas.height);
  ctx.save(); ctx.scale(dpr, dpr);

  const timeRulerH = 18;
  const midY = timeRulerH + (H - timeRulerH) / 2;
  const halfH = (H - timeRulerH - 14) / 2;
  const dur = curDur || 1;
  const curT = mainAudio.currentTime || 0;
  const startS = parseFloat(startInp.value) || 0;
  const endS   = parseFloat(endInp.value)   || Math.min(0.65, dur);

  /* ── Time Ruler (seconds ticks) ── */
  ctx.strokeStyle = "#1e293b"; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(padX, timeRulerH); ctx.lineTo(padX + audioW, timeRulerH); ctx.stroke();

  let tickStep = 1.0;
  if (audioW / dur >= 150) tickStep = 0.25;
  else if (audioW / dur >= 75) tickStep = 0.5;
  else if (audioW / dur <= 25) tickStep = 2.0;

  ctx.fillStyle = "#64748b";
  ctx.font = "9px monospace";
  ctx.textAlign = "center";
  for (let t = 0; t <= dur + 0.001; t += tickStep) {
    const tx = padX + (t / dur) * audioW;
    ctx.strokeStyle = "#334155";
    ctx.beginPath(); ctx.moveTo(tx, timeRulerH - 6); ctx.lineTo(tx, timeRulerH); ctx.stroke();
    ctx.fillText(`${t.toFixed(t % 1 === 0 ? 0 : 1)}s`, tx, timeRulerH - 8);
  }

  /* ── Baseline Horizontal Center Axis ── */
  ctx.strokeStyle = '#1e293b'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(padX, midY); ctx.lineTo(padX + audioW, midY); ctx.stroke();

  /* ── Segment Selection Tint & Boundaries ── */
  const sx = padX + (startS / dur) * audioW;
  const ex = padX + (endS / dur) * audioW;
  const selW = Math.max(3, ex - sx);

  ctx.fillStyle = "rgba(56,189,248,0.14)";
  ctx.fillRect(sx, timeRulerH, selW, H - timeRulerH);

  /* Boundary vertical lines */
  ctx.strokeStyle = "#0ea5e9"; ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(sx, timeRulerH); ctx.lineTo(sx, H);
  ctx.moveTo(ex, timeRulerH); ctx.lineTo(ex, H);
  ctx.stroke();

  // Boundary top flags
  ctx.fillStyle = "#0284c7";
  ctx.beginPath();
  ctx.moveTo(sx, timeRulerH); ctx.lineTo(sx + 5, timeRulerH - 5); ctx.lineTo(sx, timeRulerH - 5); ctx.fill();
  ctx.beginPath();
  ctx.moveTo(ex, timeRulerH); ctx.lineTo(ex - 5, timeRulerH - 5); ctx.lineTo(ex, timeRulerH - 5); ctx.fill();

  /* ── Symmetrical Amplitude Bars ── */
  if (waveData) {
    const arr = metricMode === 'rms' ? waveData.rms : waveData.peaks;
    const n   = arr.length;
    const barSpacing = audioW / n;
    const barW = Math.max(1.5, Math.min(4.5, barSpacing - 1.0));
    ctx.lineCap = "round";
    ctx.lineWidth = barW;

    for (let i = 0; i < n; i++) {
      const x = padX + i * barSpacing + barW * 0.5;
      const t = (i / n) * dur;
      const v = arr[i];
      const bh = Math.max(2.5, v * halfH);
      const played   = t <= curT;
      const selected = t >= startS && t <= endS;

      if (played) {
        ctx.strokeStyle = selected ? "#38bdf8" : "#0284c7";
      } else {
        ctx.strokeStyle = selected ? "#94a3b8" : "#334155";
      }

      ctx.beginPath();
      ctx.moveTo(x, midY - bh);
      ctx.lineTo(x, midY + bh);
      ctx.stroke();
    }

    /* ── Strum Attack Onsets (Golden Dots) ── */
    if (waveData.onsets) {
      ctx.fillStyle = "#f59e0b";
      waveData.onsets.forEach(t => {
        if (t <= dur) {
          const ox = padX + (t / dur) * audioW;
          ctx.beginPath();
          ctx.arc(ox, timeRulerH + 4, 3, 0, Math.PI * 2);
          ctx.fill();
        }
      });
    }
  }

  /* ── Playhead Needle (when Center Guide is OFF) ── */
  if (!autoCenterPlayhead) {
    const px = padX + (curT / dur) * audioW;
    ctx.shadowColor = "#38bdf8"; ctx.shadowBlur = 8;
    ctx.strokeStyle = "#ffffff"; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(px, timeRulerH); ctx.lineTo(px, H); ctx.stroke();
    ctx.shadowBlur = 0;
    ctx.fillStyle = "#38bdf8";
    ctx.beginPath(); ctx.arc(px, midY, 4.5, 0, Math.PI * 2); ctx.fill();
  }

  ctx.restore();
}

/* ── Smooth Playhead Centering & Animation Loop ── */
function syncPlayheadScroll() {
  if (!autoCenterPlayhead) return;
  const { audioW } = getWaveGeometry();
  const dur = curDur || 1;
  const frac = (mainAudio.currentTime || 0) / dur;
  waveScroll.scrollLeft = frac * audioW;
}

function onAnimationTick() {
  if (!mainAudio.paused && !mainAudio.ended) {
    const t = mainAudio.currentTime;
    curTimeDis.textContent = `${t.toFixed(2)} / ${curDur.toFixed(2)}s`;
    drawWave();
    syncPlayheadScroll();

    // Check segment boundary for looping / stop
    const e = parseFloat(endInp.value);
    if (t >= e) {
      if (isLooping) {
        mainAudio.currentTime = parseFloat(startInp.value);
        mainAudio.play();
      } else {
        mainAudio.pause();
      }
    }
    animFrameId = requestAnimationFrame(onAnimationTick);
  }
}

mainAudio.addEventListener("play", () => {
  if (animFrameId) cancelAnimationFrame(animFrameId);
  animFrameId = requestAnimationFrame(onAnimationTick);
});

mainAudio.addEventListener("pause", () => {
  if (animFrameId) cancelAnimationFrame(animFrameId);
  drawWave();
  syncPlayheadScroll();
});

mainAudio.addEventListener("seeked", () => {
  drawWave();
  syncPlayheadScroll();
});

/* ── Zoom Buttons ── */
function updateZoomButtons() {
  ["zoomFitBtn","zoom1xBtn","zoom2xBtn","zoom4xBtn"].forEach(id => {
    Q(id).className = "wave-zoom-btn bg-slate-800 text-slate-300";
  });
  const map = { fit:"zoomFitBtn", "1x":"zoom1xBtn", "2x":"zoom2xBtn", "4x":"zoom4xBtn" };
  if (map[zoomMode]) Q(map[zoomMode]).className = "wave-zoom-btn bg-sky-600 text-white";
}

function setZoom(mode) {
  zoomMode = mode;
  updateZoomButtons();
  drawWave();
  syncPlayheadScroll();
}

Q("zoomFitBtn").addEventListener("click", () => setZoom("fit"));
Q("zoom1xBtn").addEventListener("click", () => setZoom("1x"));
Q("zoom2xBtn").addEventListener("click", () => setZoom("2x"));
Q("zoom4xBtn").addEventListener("click", () => setZoom("4x"));

Q("toggleMetricBtn").addEventListener("click", () => {
  metricMode = metricMode === 'peaks' ? 'rms' : 'peaks';
  Q("toggleMetricBtn").textContent = metricMode === 'peaks' ? 'Peak' : 'RMS';
  Q("toggleMetricBtn").classList.toggle("text-sky-400", metricMode === 'rms');
  drawWave();
});

Q("centerPlayheadToggle").addEventListener("click", () => {
  autoCenterPlayhead = !autoCenterPlayhead;
  Q("centerPlayheadToggle").textContent = `Center Playhead: ${autoCenterPlayhead ? 'ON' : 'OFF'}`;
  drawWave();
  syncPlayheadScroll();
});

/* ── String Guides Toggle ── */
toggleGuides.addEventListener("click", () => {
  showGuides = !showGuides;
  toggleGuides.innerHTML = `🎸 String Guides: ${showGuides ? 'ON' : 'OFF'}`;
  if (showGuides) {
    toggleGuides.className = "px-3 py-1.5 text-xs font-semibold rounded-lg border border-sky-600/80 bg-sky-950/80 text-sky-300 hover:bg-sky-900/60 transition flex items-center gap-1.5";
    legend.classList.remove("opacity-25");
  } else {
    toggleGuides.className = "px-3 py-1.5 text-xs font-semibold rounded-lg border border-slate-700 bg-slate-800 text-slate-400 hover:bg-slate-700 transition flex items-center gap-1.5";
    legend.classList.add("opacity-25");
  }
  refreshSpec();
});

/* ── Waveform click & drag interaction ── */
function canvasEventToSec(e) {
  const rect = waveCanvas.getBoundingClientRect();
  const clickX = e.clientX - rect.left;
  const { audioW, padX } = getWaveGeometry();
  const audioX = clickX - padX;
  return Math.max(0, Math.min(curDur, (audioX / audioW) * curDur));
}

waveCanvas.addEventListener("mousedown", e => {
  if (!curDur) return;
  const sec = canvasEventToSec(e);
  if (e.shiftKey) {
    endInp.value = sec.toFixed(2);
    endSlider.value = sec;
    syncFromInputs();
  } else {
    isDragging = true;
    dragStartSec = sec;
    mainAudio.currentTime = sec;
    startInp.value = sec.toFixed(2);
    startSlider.value = sec;
    drawWave();
    syncPlayheadScroll();
  }
});

window.addEventListener("mousemove", e => {
  if (!isDragging || !curDur) return;
  const sec = canvasEventToSec(e);
  const s = Math.min(dragStartSec, sec);
  const en = Math.max(dragStartSec, sec);
  startInp.value = s.toFixed(2);
  startSlider.value = s;
  endInp.value = Math.max(s + 0.05, en).toFixed(2);
  endSlider.value = Math.max(s + 0.05, en);
  syncFromSliders(false);
  drawWave();
});

window.addEventListener("mouseup", () => {
  if (isDragging) {
    isDragging = false;
    syncFromSliders(true);
  }
});

/* Mouse wheel horizontal scroll on waveform */
waveScroll.addEventListener("wheel", e => {
  if (Math.abs(e.deltaX) > 0 || Math.abs(e.deltaY) > 0) {
    waveScroll.scrollLeft += (e.deltaX || e.deltaY);
    e.preventDefault();
  }
}, { passive: false });

/* ── Sync helpers ── */
let specDebounce = null;
function refreshSpec() {
  clearTimeout(specDebounce);
  specDebounce = setTimeout(() => {
    if (!curFile) return;
    specStatus.textContent = "Updating…";
    fetchSpectrogram(curFile, parseFloat(startSlider.value), parseFloat(endSlider.value));
  }, 400);
}

function syncFromSliders(refreshSpectrogram = true) {
  let s = parseFloat(startSlider.value), e = parseFloat(endSlider.value);
  if (s >= e) { e = Math.min(curDur, s + 0.1); endSlider.value = e; }
  startInp.value = s.toFixed(2); endInp.value = e.toFixed(2);
  startLbl.textContent = `${s.toFixed(2)}s`; endLbl.textContent = `${e.toFixed(2)}s`;
  durLbl.textContent = `${Math.max(0, e - s).toFixed(2)}s`;
  drawWave();
  if (refreshSpectrogram) refreshSpec();
}

function syncFromInputs() {
  let s = parseFloat(startInp.value) || 0, e = parseFloat(endInp.value) || 0.5;
  if (e <= s) e = s + 0.1;
  startSlider.value = s; endSlider.value = e;
  startLbl.textContent = `${s.toFixed(2)}s`; endLbl.textContent = `${e.toFixed(2)}s`;
  durLbl.textContent = `${(e - s).toFixed(2)}s`;
  drawWave(); refreshSpec();
}

startSlider.addEventListener("input", () => syncFromSliders(true));
endSlider.addEventListener("input",   () => syncFromSliders(true));
startInp.addEventListener("change", syncFromInputs);
endInp.addEventListener("change",   syncFromInputs);
audioSel.addEventListener("change", () => loadAudio(audioSel.value));
Q("selectAllBtn").addEventListener("click", () => {
  startSlider.value = 0; endSlider.value = curDur; syncFromSliders(true);
});

/* ── Audio timeupdate fallback ── */
mainAudio.addEventListener("timeupdate", () => {
  const t = mainAudio.currentTime;
  curTimeDis.textContent = `${t.toFixed(2)} / ${curDur.toFixed(2)}s`;
  drawWave();
  syncPlayheadScroll();
  const e = parseFloat(endInp.value);
  if (t >= e) {
    if (isLooping) { mainAudio.currentTime = parseFloat(startInp.value); mainAudio.play(); }
    else mainAudio.pause();
  }
});

Q("playSegmentBtn").addEventListener("click", () => {
  mainAudio.currentTime = parseFloat(startInp.value); mainAudio.play();
});
Q("loopSegmentBtn").addEventListener("click", () => {
  isLooping = !isLooping;
  Q("loopSegmentBtn").classList.toggle("bg-sky-600", isLooping);
  Q("loopSegmentBtn").classList.toggle("text-white", isLooping);
});

/* ── File upload ── */
Q("fileUploadInput").addEventListener("change", async ev => {
  const f = ev.target.files[0]; if (!f) return;
  const fd = new FormData(); fd.append("audio", f);
  const r = await fetch("/api/upload", { method:"POST", body: fd });
  const d = await r.json();
  if (d.path) {
    const o = document.createElement("option");
    o.value = d.path; o.textContent = `[Uploaded] ${d.filename}`;
    audioSel.insertBefore(o, audioSel.firstChild);
    audioSel.value = d.path; loadAudio(d.path);
  }
});

window.addEventListener("resize", drawWave);

/* ── Fretboard SVG ── */
function renderFretboard(tab4321) {
  const frets=5, W=280, H=180, fS=40, startX=60, startY=28;
  let svg = `<svg viewBox="0 0 ${W} ${H}" class="w-full max-w-xs">`;
  svg += `<line x1="${startX}" y1="${startY}" x2="${startX}" y2="${startY+3*fS}" stroke="#cbd5e1" stroke-width="6" stroke-linecap="round"/>`;
  for (let f=1;f<=frets;f++){
    const x=startX+f*fS;
    svg+=`<line x1="${x}" y1="${startY}" x2="${x}" y2="${startY+3*fS}" stroke="#475569" stroke-width="2"/>`;
    svg+=`<text x="${x-fS/2}" y="${startY+3*fS+20}" fill="#94a3b8" font-size="10" text-anchor="middle" font-family="monospace">F${f}</text>`;
  }
  ["G(4)","C(3)","E(2)","A(1)"].forEach((name,si)=>{
    const y=startY+si*fS;
    svg+=`<text x="${startX-10}" y="${y+4}" fill="#94a3b8" font-size="10" text-anchor="end">${name}</text>`;
    svg+=`<line x1="${startX}" y1="${y}" x2="${startX+frets*fS}" y2="${y}" stroke="#64748b" stroke-width="2"/>`;
    const fn=tab4321[si];
    if(fn===0){
      svg+=`<circle cx="${startX-5}" cy="${y}" r="4.5" fill="none" stroke="#38bdf8" stroke-width="2"/>`;
    } else if(fn>0&&fn<=frets){
      const dx=startX+fn*fS-fS/2;
      svg+=`<circle cx="${dx}" cy="${y}" r="7" fill="#38bdf8" stroke="#0284c7" stroke-width="1.5"/>`;
      svg+=`<text x="${dx}" y="${y+4}" fill="#0f172a" font-size="9" font-weight="bold" text-anchor="middle">${fn}</text>`;
    }
  });
  svg+="</svg>";
  Q("fretboardSvgContainer").innerHTML=svg;
}

/* ── Transcribe Segment ── */
Q("runChordBtn").addEventListener("click", async () => {
  if (!curFile) return;
  Q("runChordBtn").disabled=true;
  Q("runChordBtn").innerHTML="⏳ Analysing…";
  try {
    const r = await fetch("/api/transcribe_segment",{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({file:curFile,start_sec:+startInp.value,end_sec:+endInp.value,notes:4,top_n:6,iter:25})});
    const d = await r.json();
    if(d.error) throw new Error(d.error);
    resultsSection.classList.remove("hidden"); singleCont.classList.remove("hidden"); seqCont.classList.add("hidden");
    Q("acousticMatchBadge").textContent=`${d.acoustic_match}%`;
    Q("stftLossBadge").textContent=d.stft_loss;
    Q("tabStringDisplay").textContent=`[ G:${d.tab_4321[0]} C:${d.tab_4321[1]} E:${d.tab_4321[2]} A:${d.tab_4321[3]} ]`;
    Q("chordNotesDisplay").textContent=d.notes.join(", ");
    Q("elapsedDisplay").textContent=`${d.elapsed}s`;
    Q("combinedScoreDisplay").textContent=`${d.combined_score}%`;
    renderFretboard(d.tab_4321);
    if(d.synth_url) synthAudio.src=d.synth_url;
    Q("candidatesTableBody").innerHTML="";
    d.candidates.forEach(c=>{
      const tr=document.createElement("tr");
      const win=c.verdict==="WINNER"||c.verdict==="SELECTED";
      tr.className=win?"bg-sky-950/40 text-sky-200 font-medium":"hover:bg-slate-800/40";
      tr.innerHTML=`<td class="py-1.5 px-3 font-mono font-bold">${c.rank}</td><td class="py-1.5 px-3">${c.notes||c.desc}</td><td class="py-1.5 px-3 font-mono">${c.tab||"--"}</td><td class="py-1.5 px-3 text-emerald-400">${c.audio_match}%</td><td class="py-1.5 px-3 text-indigo-300">${c.spectral_fit||c.physics_fit||"--"}%</td><td class="py-1.5 px-3"><span class="px-1.5 py-0.5 rounded text-[10px] ${win?"bg-sky-500/20 text-sky-400 border border-sky-600/40":"text-slate-500"}">${c.verdict}</span></td>`;
      Q("candidatesTableBody").appendChild(tr);
    });
    resultsSection.scrollIntoView({behavior:"smooth"});
  } catch(e){ alert("Transcription failed: "+e.message); }
  finally { Q("runChordBtn").disabled=false; Q("runChordBtn").innerHTML="⚡ Transcribe Segment (Single Chord)"; }
});

/* ── Transcribe Sequence ── */
Q("runSequenceBtn").addEventListener("click", async () => {
  if (!curFile) return;
  Q("runSequenceBtn").disabled=true;
  Q("runSequenceBtn").innerHTML="⏳ Segmenting…";
  try {
    const r = await fetch("/api/transcribe_sequence",{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({file:curFile,start_sec:+startInp.value,end_sec:+endInp.value,min_duration:"auto",split_strums:false})});
    const d = await r.json();
    if(d.error) throw new Error(d.error);
    resultsSection.classList.remove("hidden"); singleCont.classList.add("hidden"); seqCont.classList.remove("hidden");
    Q("progressionFlowDisplay").textContent=d.progression_flow||"No chords detected.";
    if(d.synth_url) seqSynthAudio.src=d.synth_url;
    Q("sequenceTableBody").innerHTML="";
    d.segments.forEach(s=>{
      const tr=document.createElement("tr"); tr.className="hover:bg-slate-800/40";
      tr.innerHTML=`<td class="py-2 px-3 font-bold text-sky-400">#${s.index}</td><td class="py-2 px-3 font-mono">${s.start}–${s.end}s</td><td class="py-2 px-3 font-mono">${s.duration}s</td><td class="py-2 px-3 font-bold text-indigo-300">${s.chord} <span class="text-[10px] text-slate-400 font-normal">(${s.quality})</span></td><td class="py-2 px-3 font-mono text-sky-300">${s.tab_4321.join(" - ")}</td><td class="py-2 px-3 text-slate-400">[${s.notes.join(", ")}]</td><td class="py-2 px-3">${s.strum_count}</td><td class="py-2 px-3 text-emerald-400 font-semibold">${s.confidence}%</td>`;
      Q("sequenceTableBody").appendChild(tr);
    });
    resultsSection.scrollIntoView({behavior:"smooth"});
  } catch(e){ alert("Sequence failed: "+e.message); }
  finally { Q("runSequenceBtn").disabled=false; Q("runSequenceBtn").innerHTML="🎼 Transcribe Progression (Sequence)"; }
});

init();
</script>
</body>
</html>
"""


def start_ui_server(host: str = "127.0.0.1", port: int = 8080):
    """Start the multi-threaded HTTP UI server, auto-incrementing port if needed."""
    httpd = None
    for attempt in range(10):
        try:
            httpd = ThreadedHTTPServer((host, port + attempt), UIRequestHandler)
            port = port + attempt
            break
        except OSError as e:
            if attempt == 9:
                raise e

    print("\n" + "=" * 80)
    print("      TabsGeneration: Interactive Ukulele Audio & Tab Inspector")
    print("=" * 80)
    print(f"   Local UI Server URL:  http://{host}:{port}/")
    print(f"   Audio Files Scanned:  {len(get_available_audio_files())} files detected")
    print(f"   Hardware Engine:      {torch.cuda.get_device_name(DEVICE) if DEVICE.type == 'cuda' else 'CPU'}")
    print("=" * 80)
    print("   Press Ctrl+C to stop the server.\n")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[TabsGeneration UI] Server stopped.")
    finally:
        if httpd:
            httpd.server_close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="TabsGeneration Interactive Web UI Server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    start_ui_server(host=args.host, port=args.port)
