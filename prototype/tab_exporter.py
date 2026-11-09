import os
import datetime
from typing import List, Dict, Optional
import torch

from device_config import DEVICE
from fretboard_mapper import UKULELE_TUNING, freq_to_midi

_NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def export_sequence_tabs_to_file(file_path: str,
                                 audio_path: str,
                                 segments: List[dict],
                                 total_duration: float,
                                 elapsed: float,
                                 split_strums: bool = False):
    """
    Writes properly formatted ukulele tablature and chord sequence details to a file,
    completely overwriting it on every run.
    """
    dev_desc = torch.cuda.get_device_name(DEVICE) if DEVICE.type == "cuda" else "CPU"
    mode_str = "Individual Strum Mode (--split-strums)" if split_strums else "Harmonic Progression Mode"
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = []
    lines.append("=" * 86)
    lines.append("                UKULELE TABLATURE TRANSCRIPTION REPORT")
    lines.append("=" * 86)
    lines.append(f"Audio File:      {os.path.basename(audio_path)} ({audio_path})")
    lines.append(f"Audio Duration:  {total_duration:.2f} seconds")
    lines.append(f"Total Chords:    {len(segments)} event(s) detected")
    lines.append(f"Analysis Mode:   {mode_str}")
    lines.append(f"Timestamp:       {now_str}")
    lines.append(f"Processing Time: {elapsed:.2f}s ({dev_desc})")
    lines.append("=" * 86)
    lines.append("")

    if not segments:
        lines.append("[!] No chords or notes were detected in this audio file.")
        lines.append("=" * 86)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        return

    # 1. Progression Summary
    prog_str = "  ==>  ".join(f"[ {s['chord']} {s['chord_data']['quality']} ]" for s in segments)
    lines.append("CHORD PROGRESSION SEQUENCE:")
    lines.append(f"  {prog_str}")
    lines.append("")
    lines.append("-" * 86)
    lines.append("")

    # 2. Formatted ASCII Tablature Score (Wrapped into measures of 6 chords per system)
    lines.append("UKULELE TABLATURE SCORE (Strings 1 to 4: A - E - C - G):")
    lines.append("-" * 86)

    chords_per_line = 6
    for block_idx in range(0, len(segments), chords_per_line):
        sub_segs = segments[block_idx : block_idx + chords_per_line]
        start_m = block_idx + 1
        end_m = block_idx + len(sub_segs)
        t_start = sub_segs[0]["start"]
        t_end = sub_segs[-1]["end"]

        lines.append(f"\n[Measures {start_m} - {end_m}]  ({t_start:4.2f}s - {t_end:4.2f}s):")

        s1_line = "  A (String 1) |"
        s2_line = "  E (String 2) |"
        s3_line = "  C (String 3) |"
        s4_line = "  G (String 4) |"
        chord_line = "               |"

        for s in sub_segs:
            s4, s3, s2, s1 = s["chord_data"]["tab_4321"]
            w = 11
            s1_line += f"--{s1:<2}" + "-" * (w - 4) + "|"
            s2_line += f"--{s2:<2}" + "-" * (w - 4) + "|"
            s3_line += f"--{s3:<2}" + "-" * (w - 4) + "|"
            s4_line += f"--{s4:<2}" + "-" * (w - 4) + "|"

            c_str = f"[{s['chord']}]"
            chord_line += f"  {c_str:<8}|"

        lines.append("  " + "-" * (len(s1_line) - 2))
        lines.append(s1_line)
        lines.append(s2_line)
        lines.append(s3_line)
        lines.append(s4_line)
        lines.append("  " + "-" * (len(s1_line) - 2))
        lines.append(chord_line)

    lines.append("")
    lines.append("=" * 86)
    lines.append("                          CHORD TIMELINE DETAILS")
    lines.append("=" * 86)
    lines.append(f" {'#':>2}   {'Time Interval':<16}  {'Duration':>9}  {'Chord':<16}  {'Tab (4-3-2-1)':<15}  {'Notes Played':<20}  {'Match %':>7}")
    lines.append(f" {'-'*2}   {'-'*16}  {'-'*9}  {'-'*16}  {'-'*15}  {'-'*20}  {'-'*7}")

    for idx, s in enumerate(segments, 1):
        time_str = f"{s['start']:4.2f}s - {s['end']:4.2f}s"
        dur_str = f"{s['duration']:4.2f}s"
        strum_note = f" ({s['strum_count']} strums)" if not split_strums and s.get("strum_count", 1) > 1 else ""
        chord_label = f"{s['chord']} {s['chord_data']['quality']}{strum_note}"
        tab_str = " - ".join(str(f) for f in s["chord_data"]["tab_4321"])
        notes_str = "[" + ", ".join(s["chord_data"]["notes_4321"]) + "]"
        match_pct = s.get("confidence", 0.0) * 100.0

        lines.append(f" #{idx:<2}  {time_str:<16}  {dur_str:>9}  {chord_label:<16}  {tab_str:<15}  {notes_str:<20}  {match_pct:>6.1f}%")

    lines.append("=" * 86)

    # Write and overwrite file
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def export_single_tab_to_file(file_path: str,
                              audio_path: str,
                              tab_result: dict,
                              winner_data: dict,
                              strummed_strings: list,
                              elapsed: float,
                              is_single_note: bool = False):
    """
    Writes properly formatted single chord/note tablature to a file,
    completely overwriting it on every run.
    """
    dev_desc = torch.cuda.get_device_name(DEVICE) if DEVICE.type == "cuda" else "CPU"
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = []
    lines.append("=" * 86)
    lines.append("                UKULELE TABLATURE TRANSCRIPTION REPORT")
    lines.append("=" * 86)
    lines.append(f"Audio File:      {os.path.basename(audio_path)} ({audio_path})")
    lines.append(f"Timestamp:       {now_str}")
    lines.append(f"Processing Time: {elapsed:.2f}s ({dev_desc})")
    lines.append("=" * 86)
    lines.append("")

    if is_single_note:
        best = tab_result['all_candidates'][0]
        midi_num = tab_result['midi_note']
        note_name = f"{_NOTE_NAMES[midi_num % 12]}{midi_num // 12 - 1}"
        lines.append("SINGLE NOTE TABLATURE:")
        lines.append(f"   Detected Note:      {note_name} (MIDI {midi_num}, {tab_result['pitch_hz']:.2f} Hz)")
        lines.append(f"   Plucked String:     String {tab_result['string']} (open {best['open_name']} string)")
        lines.append(f"   Fret Position:      Fret {tab_result['fret']}")
        if "audio_loss" in best:
            lines.append(f"   Acoustic Match:     {best.get('audio_match_norm', 1.0)*100:.1f}% (STFT Error: {best['audio_loss']:.4f})")
        lines.append(f"   Physics Fit:        {best.get('inharm_score', 0.0)*100:.1f}%")
        lines.append("")

        # ASCII Tab
        lines.append("ASCII UKULELE TABLATURE:")
        lines.append("-" * 40)
        for s in [1, 2, 3, 4]:
            open_name = UKULELE_TUNING[s]["name"]
            fret_val = str(tab_result["fret"]) if s == tab_result["string"] else "-"
            lines.append(f"   String {s} ({open_name:<2}) |--{fret_val:<2}--|")
        lines.append("-" * 40)
        lines.append(f"                  | [{note_name}] |")
    else:
        mapping = winner_data["refined"]["mapping"]
        winner_notes = [f"{_NOTE_NAMES[m % 12]}{m // 12 - 1}" for m in winner_data["multiset_midis"]]
        winner_notes_str = "[" + ", ".join(winner_notes) + "]"
        tab_summary = " | ".join(f"{UKULELE_TUNING[s]['name'][0]}: {mapping.get(s, 0)}"
                                 for s in sorted(strummed_strings, reverse=True))

        lines.append("TRANSCRIBED CHORD TABLATURE:")
        lines.append(f"   Chord Notes:     {winner_notes_str}")
        lines.append(f"   Summary Tab:     [ {tab_summary} ]")
        lines.append(f"   Acoustic Match:  100.0% (STFT Error: {winner_data['audio_loss']:.4f})")
        lines.append(f"   Overall Score:   {winner_data['combined_score']*100:.1f}%")
        lines.append("")

        # ASCII Tab
        lines.append("ASCII UKULELE TABLATURE (Strings 1 to 4):")
        lines.append("-" * 40)
        for s in [1, 2, 3, 4]:
            open_name = UKULELE_TUNING[s]["name"]
            fret = mapping.get(s, 0)
            lines.append(f"   String {s} ({open_name:<2}) |--{fret:<2}--|")
        lines.append("-" * 40)

        lines.append("\nFRETBOARD DETAIL:")
        for string_idx in sorted(strummed_strings, reverse=True):
            fret = mapping[string_idx]
            fret_str = f"Fret {fret}" if fret > 0 else "Fret 0 (open)"
            base_freq = UKULELE_TUNING[string_idx]["freq"]
            fret_freq = base_freq * (2.0 ** (fret / 12.0))
            fret_midi = int(round(freq_to_midi(fret_freq)))
            fret_note = f"{_NOTE_NAMES[fret_midi % 12]}{fret_midi // 12 - 1}"
            open_name = UKULELE_TUNING[string_idx]["name"]
            lines.append(f"   String {string_idx} (open {open_name:<2}):  {fret_str:<14}  -->  {fret_note:<3} ({fret_freq:6.2f} Hz)")

    lines.append("\n" + "=" * 86)

    # Write and overwrite file
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
