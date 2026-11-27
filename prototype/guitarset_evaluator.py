"""
GuitarSet Comping Excerpt Benchmark Evaluator
Implements the full checklist for 6-string guitar transcription:
1. Data: 180 comping excerpts (mic or mix audio). Players kept separate (tune vs test).
2. Model setup: 6 strings, standard tuning (EADGBE), steel-string parameters.
3. Ground truth: Per-string notes from JAMS (onset, duration, pitch, string, fret).
4. Matching rule: 50 ms onset window, pitch match; string match for string-aware score.
5. Metrics: Precision, Recall, F1 for Pitch-only and Pitch+String; F1 position gap.
6. One-to-one matching: Greedy bipartite matching preventing repeated prediction inflation.
7. Breakdown: Reported by Player, Musical Style, Tempo, and Chord Progression.
8. Limits: 3 chord progressions only, 6-string guitar.
9. Dry run: Tested on single excerpt first.
"""

import os
import sys
import glob
import json
import time
import re
import argparse
from typing import List, Dict, Tuple, Any, Optional
import numpy as np

PROTOTYPE_DIR = os.path.dirname(os.path.abspath(__file__))
if PROTOTYPE_DIR not in sys.path:
    sys.path.insert(0, PROTOTYPE_DIR)

# Standard guitar open string MIDI pitches (EADGBE):
# String 6: E2 = 40, String 5: A2 = 45, String 4: D3 = 50,
# String 3: G3 = 55, String 2: B3 = 59, String 1: E4 = 64
OPEN_MIDIS = {
    6: 40,
    5: 45,
    4: 50,
    3: 55,
    2: 59,
    1: 64
}

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def parse_excerpt_filename(filepath: str) -> Dict[str, Any]:
    """
    Parse filename like '00_BN1-129-Eb_comp.jams' into metadata components.
    """
    basename = os.path.basename(filepath)
    name_no_ext = os.path.splitext(basename)[0]
    name_clean = re.sub(r'_(mic|mix)$', '', name_no_ext)
    
    match = re.match(r'^(\d+)_([A-Za-z]+)(\d+)-(\d+)-([A-Za-z0-9#]+)_([a-z]+)', name_clean)
    if match:
        player, style, prog, tempo, key, mode = match.groups()
        return {
            "id": name_clean,
            "player": player,
            "style": style,
            "progression": int(prog),
            "tempo": int(tempo),
            "key": key,
            "mode": mode,
            "basename": name_clean
        }
    return {
        "id": name_clean,
        "player": "00",
        "style": "BN",
        "progression": 1,
        "tempo": 120,
        "key": "C",
        "mode": "comp",
        "basename": name_clean
    }


def load_guitarset_ground_truth(jams_path: str) -> List[Dict[str, Any]]:
    """
    Load ground truth note events from a GuitarSet JAMS file.
    Extracts per-string notes from 'note_midi' annotations.
    String numbers are 1-based: 6 (low E) to 1 (high E).
    Fret = pitch - open_string_midi.
    """
    with open(jams_path, "r", encoding="utf-8") as f:
        jams_data = json.load(f)

    gt_notes = []
    for a in jams_data.get("annotations", []):
        if a.get("namespace") == "note_midi":
            meta = a.get("annotation_metadata", {})
            src = meta.get("data_source", None)
            if src is not None:
                try:
                    src_int = int(src)
                    string_num = 6 - src_int
                except ValueError:
                    continue

                open_m = OPEN_MIDIS.get(string_num, 40)
                for obs in a.get("data", []):
                    onset = float(obs["time"])
                    dur = float(obs["duration"])
                    val = float(obs["value"])
                    pitch = int(round(val))
                    fret = pitch - open_m

                    gt_notes.append({
                        "onset": onset,
                        "offset": onset + dur,
                        "duration": dur,
                        "pitch": pitch,
                        "string": string_num,
                        "fret": fret,
                        "exact_pitch": val
                    })

    gt_notes.sort(key=lambda n: (n["onset"], n["string"]))
    return gt_notes


def allocate_notes_to_guitar_strings(
    note_events: List[Tuple[float, float, int, float]],
    cluster_window: float = 0.040
) -> List[Dict[str, Any]]:
    """
    Allocate predicted note events to 6-string guitar fretboard.
    Uses position-aware clustering and ergonomic hand span optimization.
    """
    sorted_events = sorted(note_events, key=lambda x: x[0])
    
    # 1. Cluster events by onset time proximity
    clusters = []
    for onset, offset, pitch, amp in sorted_events:
        pitch = int(round(pitch))
        if not clusters:
            clusters.append([(onset, offset, pitch, amp)])
        else:
            if abs(onset - clusters[-1][0][0]) <= cluster_window:
                clusters[-1].append((onset, offset, pitch, amp))
            else:
                clusters.append([(onset, offset, pitch, amp)])

    predicted_notes = []
    hand_pos = 5  # default hand position

    for cluster in clusters:
        # Separate unique pitches in cluster
        cluster_notes = {}
        for onset, offset, pitch, amp in cluster:
            if pitch not in cluster_notes or amp > cluster_notes[pitch][3]:
                cluster_notes[pitch] = (onset, offset, pitch, amp)

        pitches_to_assign = sorted(list(cluster_notes.keys()))
        if not pitches_to_assign:
            continue

        # Find best position-aware assignment
        best_assignment, hand_pos = _allocate_cluster_position_aware(
            pitches_to_assign, prev_pos=hand_pos
        )

        for p, s in best_assignment.items():
            onset, offset, _, amp = cluster_notes[p]
            fret = p - OPEN_MIDIS[s]
            predicted_notes.append({
                "onset": onset,
                "offset": offset,
                "duration": offset - onset,
                "pitch": p,
                "string": s,
                "fret": fret,
                "amplitude": amp
            })

    predicted_notes.sort(key=lambda n: (n["onset"], n["string"]))
    return predicted_notes


def _allocate_cluster_position_aware(
    pitches: List[int],
    prev_pos: int = 5
) -> Tuple[Dict[int, int], int]:
    """
    Search over candidate hand positions (0..11) to find the assignment
    that minimizes fret span and movement from previous hand position.
    """
    if not pitches:
        return {}, prev_pos

    positions_to_try = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    best_cost = float('inf')
    best_assignment = {}
    best_pos = prev_pos

    for pos in positions_to_try:
        pos_strings = {}
        for p in pitches:
            valid_s = []
            for s in [6, 5, 4, 3, 2, 1]:
                fret = p - OPEN_MIDIS[s]
                # Allowed if open string (0) or within hand reach [pos, pos+4]
                if fret == 0 or (pos <= fret <= pos + 4 and fret <= 19):
                    valid_s.append(s)
            if not valid_s:
                valid_s = [s for s in [6, 5, 4, 3, 2, 1] if 0 <= (p - OPEN_MIDIS[s]) <= 19]
            if not valid_s:
                valid_s = [6] if p < 40 else [1]
            pos_strings[p] = valid_s

        curr_best_cost = [float('inf')]
        curr_best_assign = [{}]

        def bt(idx, assign, used):
            if idx == len(pitches):
                frets = [pitches[i] - OPEN_MIDIS[assign[pitches[i]]] for i in range(len(pitches))]
                non_zero = [f for f in frets if f > 0]
                span = (max(non_zero) - min(non_zero)) if len(non_zero) > 1 else 0
                
                # Penalize non-monotonic string order (lower pitch on physically higher string)
                mono = 0.0
                for i in range(len(pitches) - 1):
                    if assign[pitches[i]] < assign[pitches[i+1]]:
                        mono += 50.0

                pos_move = abs(pos - prev_pos) * 1.5
                c = span * 10.0 + mono + pos_move + sum(frets) * 0.1
                if c < curr_best_cost[0]:
                    curr_best_cost[0] = c
                    curr_best_assign[0] = dict(assign)
                return

            p = pitches[idx]
            cands = [s for s in pos_strings[p] if s not in used]
            if not cands:
                cands = [pos_strings[p][0]]
            for s in cands:
                assign[p] = s
                bt(idx + 1, assign, used | {s})
                del assign[p]

        bt(0, {}, frozenset())
        if curr_best_cost[0] < best_cost:
            best_cost = curr_best_cost[0]
            best_assignment = curr_best_assign[0]
            best_pos = pos

    if not best_assignment:
        best_assignment = {p: ([s for s in [6, 5, 4, 3, 2, 1] if 0 <= (p - OPEN_MIDIS[s]) <= 19] or [1])[0] for p in pitches}

    return best_assignment, best_pos


def evaluate_notes_one_to_one(
    gt_notes: List[Dict[str, Any]],
    pred_notes: List[Dict[str, Any]],
    onset_tolerance: float = 0.050
) -> Dict[str, Any]:
    """
    Greedy one-to-one bipartite matching matching rule:
    - Pitch-only: |onset_diff| <= onset_tolerance and pitch matches.
    - Pitch+string: |onset_diff| <= onset_tolerance and pitch matches and string matches.
    Computes Precision, Recall, F1 for both criteria and the F1 position gap.
    """
    n_gt = len(gt_notes)
    n_pred = len(pred_notes)

    if n_gt == 0 and n_pred == 0:
        return {
            "n_gt": 0, "n_pred": 0,
            "tp_pitch": 0, "fp_pitch": 0, "fn_pitch": 0,
            "p_pitch": 1.0, "r_pitch": 1.0, "f1_pitch": 1.0,
            "tp_string": 0, "fp_string": 0, "fn_string": 0,
            "p_string": 1.0, "r_string": 1.0, "f1_string": 1.0,
            "f1_gap": 0.0
        }
    if n_gt == 0 or n_pred == 0:
        return {
            "n_gt": n_gt, "n_pred": n_pred,
            "tp_pitch": 0, "fp_pitch": n_pred, "fn_pitch": n_gt,
            "p_pitch": 0.0, "r_pitch": 0.0, "f1_pitch": 0.0,
            "tp_string": 0, "fp_string": n_pred, "fn_string": n_gt,
            "p_string": 0.0, "r_string": 0.0, "f1_string": 0.0,
            "f1_gap": 0.0
        }

    # 1. Pitch-Only Matching
    cand_pairs_pitch = []
    for g_idx, g in enumerate(gt_notes):
        for p_idx, p in enumerate(pred_notes):
            diff = abs(g["onset"] - p["onset"])
            if diff <= onset_tolerance and g["pitch"] == p["pitch"]:
                cand_pairs_pitch.append((diff, g_idx, p_idx))

    cand_pairs_pitch.sort(key=lambda x: x[0])

    matched_gt_pitch = set()
    matched_pred_pitch = set()
    tp_pitch = 0

    for diff, g_idx, p_idx in cand_pairs_pitch:
        if g_idx not in matched_gt_pitch and p_idx not in matched_pred_pitch:
            matched_gt_pitch.add(g_idx)
            matched_pred_pitch.add(p_idx)
            tp_pitch += 1

    p_pitch = tp_pitch / n_pred if n_pred > 0 else 0.0
    r_pitch = tp_pitch / n_gt if n_gt > 0 else 0.0
    f1_pitch = (2 * p_pitch * r_pitch / (p_pitch + r_pitch)) if (p_pitch + r_pitch) > 0 else 0.0

    # 2. Pitch + String Matching
    cand_pairs_string = []
    for g_idx, g in enumerate(gt_notes):
        for p_idx, p in enumerate(pred_notes):
            diff = abs(g["onset"] - p["onset"])
            if diff <= onset_tolerance and g["pitch"] == p["pitch"] and g["string"] == p["string"]:
                cand_pairs_string.append((diff, g_idx, p_idx))

    cand_pairs_string.sort(key=lambda x: x[0])

    matched_gt_string = set()
    matched_pred_string = set()
    tp_string = 0

    for diff, g_idx, p_idx in cand_pairs_string:
        if g_idx not in matched_gt_string and p_idx not in matched_pred_string:
            matched_gt_string.add(g_idx)
            matched_pred_string.add(p_idx)
            tp_string += 1

    p_string = tp_string / n_pred if n_pred > 0 else 0.0
    r_string = tp_string / n_gt if n_gt > 0 else 0.0
    f1_string = (2 * p_string * r_string / (p_string + r_string)) if (p_string + r_string) > 0 else 0.0

    f1_gap = f1_pitch - f1_string

    return {
        "n_gt": n_gt,
        "n_pred": n_pred,
        "tp_pitch": tp_pitch,
        "fp_pitch": n_pred - tp_pitch,
        "fn_pitch": n_gt - tp_pitch,
        "p_pitch": p_pitch,
        "r_pitch": r_pitch,
        "f1_pitch": f1_pitch,
        "tp_string": tp_string,
        "fp_string": n_pred - tp_string,
        "fn_string": n_gt - tp_string,
        "p_string": p_string,
        "r_string": r_string,
        "f1_string": f1_string,
        "f1_gap": f1_gap
    }


def run_transcription_on_audio(
    audio_path: str,
    onset_thresh: float = 0.45,
    frame_thresh: float = 0.25,
    min_length: float = 80.0
) -> List[Dict[str, Any]]:
    """
    Run Spotify Basic Pitch neural inference and allocate to 6 guitar strings.
    """
    from basic_pitch.inference import predict

    model_output, midi_data, note_events = predict(
        audio_path,
        onset_threshold=onset_thresh,
        frame_threshold=frame_thresh,
        minimum_note_length=min_length
    )

    raw_events = [(float(start), float(end), int(round(pitch)), float(amp))
                  for start, end, pitch, amp, bends in note_events]

    pred_notes = allocate_notes_to_guitar_strings(raw_events)
    return pred_notes


def run_dry_run_on_excerpt(
    jams_path: str,
    audio_path: str,
    onset_tol: float = 0.050
) -> Dict[str, Any]:
    """
    Dry run evaluation on a single excerpt (Item 9 on checklist).
    """
    meta = parse_excerpt_filename(jams_path)
    print("=" * 86)
    print(f"               GUITARSET COMPING DRY RUN: {meta['id']}")
    print("=" * 86)
    print(f"JAMS File:    {jams_path}")
    print(f"Audio File:   {audio_path}")
    print(f"Metadata:     Player: {meta['player']} | Style: {meta['style']} | Prog: {meta['progression']} | Tempo: {meta['tempo']} BPM | Key: {meta['key']}")
    print("-" * 86)

    gt_notes = load_guitarset_ground_truth(jams_path)
    print(f"Parsed {len(gt_notes)} ground truth notes across 6 strings.")

    print("Running neural transcription and 6-string fretboard allocation...")
    t0 = time.time()
    pred_notes = run_transcription_on_audio(audio_path)
    elapsed = time.time() - t0
    print(f"Inference completed in {elapsed:.2f}s. Detected {len(pred_notes)} predicted notes.")

    metrics = evaluate_notes_one_to_one(gt_notes, pred_notes, onset_tolerance=onset_tol)

    print("\n" + "=" * 86)
    print(f"                  MATCHING RESULTS (50ms Onset Tolerance)")
    print("=" * 86)
    print(f"  Ground Truth Notes:        {metrics['n_gt']}")
    print(f"  Predicted Notes:           {metrics['n_pred']}")
    print("-" * 86)
    print("  [Pitch-Only Matching]:")
    print(f"    True Positives:          {metrics['tp_pitch']}")
    print(f"    False Positives:         {metrics['fp_pitch']}")
    print(f"    False Negatives:         {metrics['fn_pitch']}")
    print(f"    Precision:               {metrics['p_pitch']*100:.2f}%")
    print(f"    Recall:                  {metrics['r_pitch']*100:.2f}%")
    print(f"    F1-Score:                {metrics['f1_pitch']*100:.2f}%")
    print("-" * 86)
    print("  [Pitch + String Matching]:")
    print(f"    True Positives:          {metrics['tp_string']}")
    print(f"    False Positives:         {metrics['fp_string']}")
    print(f"    False Negatives:         {metrics['fn_string']}")
    print(f"    Precision:               {metrics['p_string']*100:.2f}%")
    print(f"    Recall:                  {metrics['r_string']*100:.2f}%")
    print(f"    F1-Score:                {metrics['f1_string']*100:.2f}%")
    print("-" * 86)
    print(f"  [Position / Fretboard Gap]: {metrics['f1_gap']*100:.2f}%  (F1_pitch - F1_pitch+string)")
    print("=" * 86)

    print("\nSample Ground Truth vs Predictions (first 10 seconds):")
    print(f" {'GT Onset':<10} {'GT Pitch':<10} {'GT Str':<8} {'GT Fret':<8} {'Status':<15} {'Matched Pred Note'}")
    print(f" {'-'*9}  {'-'*9}  {'-'*7}  {'-'*7}  {'-'*14} {'-'*20}")

    for g in gt_notes[:20]:
        g_name = f"{NOTE_NAMES[g['pitch'] % 12]}{g['pitch'] // 12 - 1}"
        matched = [p for p in pred_notes if abs(p['onset'] - g['onset']) <= onset_tol and p['pitch'] == g['pitch']]
        if matched:
            best_match = matched[0]
            m_name = f"{NOTE_NAMES[best_match['pitch'] % 12]}{best_match['pitch'] // 12 - 1}"
            str_ok = (best_match['string'] == g['string'])
            status = "MATCH (P+S)" if str_ok else "MATCH (Pitch)"
            pred_desc = f"t={best_match['onset']:.3f}s {m_name} Str {best_match['string']} Fret {best_match['fret']}"
        else:
            status = "MISSED (FN)"
            pred_desc = "-"

        print(f" {g['onset']:<10.3f} {g_name} ({g['pitch']:<2})  Str {g['string']:<3} Fret {g['fret']:<3} {status:<15} {pred_desc}")

    return {
        "metadata": meta,
        "metrics": metrics,
        "elapsed": elapsed
    }


def run_benchmark_suite(
    dataset_dir: str = "datasets/guitar",
    audio_type: str = "mix",
    limit: Optional[int] = None,
    train_players: Optional[List[str]] = None,
    test_players: Optional[List[str]] = None,
    onset_tol: float = 0.050
) -> Dict[str, Any]:
    """
    Run full GuitarSet evaluation suite over comping excerpts.
    """
    if train_players is None:
        train_players = ["00", "01", "02"]
    if test_players is None:
        test_players = ["03", "04", "05"]

    annot_dir = os.path.join(dataset_dir, "annotation")
    audio_subdir = "audio_mono-pickup_mix" if audio_type == "mix" else "audio_mono-mic"
    audio_dir = os.path.join(dataset_dir, audio_subdir)
    audio_suffix = f"_{audio_type}.wav"

    jams_pattern = os.path.join(annot_dir, "*_comp.jams")
    all_jams = sorted(glob.glob(jams_pattern))

    print(f"Discovered {len(all_jams)} comping JAMS files in '{annot_dir}'.")
    if limit is not None and limit > 0:
        # Stratify selection across all 6 players and styles
        styles_list = ["BN", "Funk", "Jazz", "Rock", "SS"]
        players_list = ["00", "01", "02", "03", "04", "05"]
        per_player = max(1, limit // len(players_list))
        
        stratified = []
        for p in players_list:
            p_jams = [j for j in all_jams if os.path.basename(j).startswith(p)]
            p_selected = []
            # First pick 1 from each style
            for st in styles_list:
                st_jams = [j for j in p_jams if parse_excerpt_filename(j)["style"] == st]
                if st_jams:
                    p_selected.append(st_jams[0])
            # If we need more or fewer for this player:
            if len(p_selected) < per_player:
                rem = [j for j in p_jams if j not in p_selected]
                p_selected.extend(rem[:per_player - len(p_selected)])
            else:
                p_selected = p_selected[:per_player]
            stratified.extend(p_selected)
        
        all_jams = stratified[:limit]
        print(f"Selected {len(all_jams)} stratified excerpts across 6 players and styles.")

    results = []

    for idx, j_path in enumerate(all_jams, 1):
        meta = parse_excerpt_filename(j_path)
        base = meta["basename"]
        audio_path = os.path.join(audio_dir, f"{base}{audio_suffix}")

        if not os.path.exists(audio_path):
            print(f"[{idx}/{len(all_jams)}] Warning: Audio not found: {audio_path}")
            continue

        split = "tune" if meta["player"] in train_players else ("test" if meta["player"] in test_players else "other")

        print(f"[{idx}/{len(all_jams)}] Eval {meta['id']} ({split.upper()}) ...", end=" ", flush=True)
        t_start = time.time()
        gt = load_guitarset_ground_truth(j_path)
        pred = run_transcription_on_audio(audio_path)
        m = evaluate_notes_one_to_one(gt, pred, onset_tolerance=onset_tol)
        el = time.time() - t_start
        print(f"Done in {el:.2f}s | F1_p: {m['f1_pitch']*100:.1f}% | F1_s: {m['f1_string']*100:.1f}% | Gap: {m['f1_gap']*100:.1f}%")

        results.append({
            "meta": meta,
            "split": split,
            "metrics": m,
            "elapsed": el
        })

    return aggregate_results(results, train_players, test_players)


def aggregate_results(results: List[Dict[str, Any]], train_players: List[str], test_players: List[str]) -> Dict[str, Any]:
    """
    Aggregate metrics across overall, train/tune split, test split, player, style, tempo, and progression.
    """
    def _calc_macro(subset):
        if not subset:
            return {"count": 0, "p_pitch": 0.0, "r_pitch": 0.0, "f1_pitch": 0.0,
                    "p_string": 0.0, "r_string": 0.0, "f1_string": 0.0, "f1_gap": 0.0,
                    "n_gt": 0, "n_pred": 0}
        
        tot_gt = sum(r["metrics"]["n_gt"] for r in subset)
        tot_pred = sum(r["metrics"]["n_pred"] for r in subset)
        tot_tp_p = sum(r["metrics"]["tp_pitch"] for r in subset)
        tot_fp_p = sum(r["metrics"]["fp_pitch"] for r in subset)
        tot_fn_p = sum(r["metrics"]["fn_pitch"] for r in subset)
        tot_tp_s = sum(r["metrics"]["tp_string"] for r in subset)
        tot_fp_s = sum(r["metrics"]["fp_string"] for r in subset)
        tot_fn_s = sum(r["metrics"]["fn_string"] for r in subset)

        p_p = tot_tp_p / (tot_tp_p + tot_fp_p) if (tot_tp_p + tot_fp_p) > 0 else 0.0
        r_p = tot_tp_p / (tot_tp_p + tot_fn_p) if (tot_tp_p + tot_fn_p) > 0 else 0.0
        f1_p = (2 * p_p * r_p / (p_p + r_p)) if (p_p + r_p) > 0 else 0.0

        p_s = tot_tp_s / (tot_tp_s + tot_fp_s) if (tot_tp_s + tot_fp_s) > 0 else 0.0
        r_s = tot_tp_s / (tot_tp_s + tot_fn_s) if (tot_tp_s + tot_fn_s) > 0 else 0.0
        f1_s = (2 * p_s * r_s / (p_s + r_s)) if (p_s + r_s) > 0 else 0.0

        return {
            "count": len(subset),
            "n_gt": tot_gt,
            "n_pred": tot_pred,
            "p_pitch": p_p,
            "r_pitch": r_p,
            "f1_pitch": f1_p,
            "p_string": p_s,
            "r_string": r_s,
            "f1_string": f1_s,
            "f1_gap": f1_p - f1_s
        }

    overall = _calc_macro(results)
    tune_split = _calc_macro([r for r in results if r["split"] == "tune"])
    test_split = _calc_macro([r for r in results if r["split"] == "test"])

    # Breakdown by Player
    by_player = {}
    players = sorted(list(set(r["meta"]["player"] for r in results)))
    for p in players:
        by_player[p] = _calc_macro([r for r in results if r["meta"]["player"] == p])

    # Breakdown by Style
    by_style = {}
    styles = sorted(list(set(r["meta"]["style"] for r in results)))
    for st in styles:
        by_style[st] = _calc_macro([r for r in results if r["meta"]["style"] == st])

    # Breakdown by Tempo
    by_tempo = {
        "Slow (<105 BPM)": _calc_macro([r for r in results if r["meta"]["tempo"] < 105]),
        "Medium (105-145 BPM)": _calc_macro([r for r in results if 105 <= r["meta"]["tempo"] < 145]),
        "Fast (>=145 BPM)": _calc_macro([r for r in results if r["meta"]["tempo"] >= 145]),
    }

    # Breakdown by Progression
    by_prog = {}
    progs = sorted(list(set(r["meta"]["progression"] for r in results)))
    for pr in progs:
        by_prog[f"Progression {pr}"] = _calc_macro([r for r in results if r["meta"]["progression"] == pr])

    return {
        "overall": overall,
        "tune_split": tune_split,
        "test_split": test_split,
        "by_player": by_player,
        "by_style": by_style,
        "by_tempo": by_tempo,
        "by_progression": by_prog,
        "excerpts": results
    }


def print_summary_report(benchmark_data: Dict[str, Any]):
    """
    Print formatted terminal report following the checklist.
    """
    print("\n" + "=" * 102)
    print("                 GUITARSET COMPING BENCHMARK EVALUATION REPORT")
    print("=" * 102)

    def _row(label, d):
        print(f" {label:<26} | {d['count']:>4} | {d['p_pitch']*100:>6.1f}% {d['r_pitch']*100:>6.1f}% {d['f1_pitch']*100:>6.1f}% | {d['p_string']*100:>6.1f}% {d['r_string']*100:>6.1f}% {d['f1_string']*100:>6.1f}% | {d['f1_gap']*100:>6.1f}%")

    print(f" {'Category / Subset':<26} | {'Ex':>4} | {'Pitch-Only (P / R / F1)':<22} | {'Pitch+String (P / R / F1)':<24} | {'F1 Gap':<7}")
    print(f" {'-'*26} | {'-'*4} | {'-'*22} | {'-'*24} | {'-'*7}")

    _row("OVERALL (Evaluated)", benchmark_data["overall"])
    _row("Tuning Split (00,01,02)", benchmark_data["tune_split"])
    _row("Test Split (03,04,05)", benchmark_data["test_split"])

    print("\n-- Breakdown by Player --")
    print(f" {'Player ID':<26} | {'Ex':>4} | {'Pitch-Only (P / R / F1)':<22} | {'Pitch+String (P / R / F1)':<24} | {'F1 Gap':<7}")
    print(f" {'-'*26} | {'-'*4} | {'-'*22} | {'-'*24} | {'-'*7}")
    for p, d in benchmark_data["by_player"].items():
        _row(f"Player {p}", d)

    print("\n-- Breakdown by Musical Style --")
    print(f" {'Style':<26} | {'Ex':>4} | {'Pitch-Only (P / R / F1)':<22} | {'Pitch+String (P / R / F1)':<24} | {'F1 Gap':<7}")
    print(f" {'-'*26} | {'-'*4} | {'-'*22} | {'-'*24} | {'-'*7}")
    style_names = {"BN": "Bossa Nova (BN)", "Funk": "Funk", "Jazz": "Jazz", "Rock": "Rock", "SS": "Singer-Songwriter (SS)"}
    for st, d in benchmark_data["by_style"].items():
        _row(style_names.get(st, st), d)

    print("\n-- Breakdown by Tempo --")
    print(f" {'Tempo Range':<26} | {'Ex':>4} | {'Pitch-Only (P / R / F1)':<22} | {'Pitch+String (P / R / F1)':<24} | {'F1 Gap':<7}")
    print(f" {'-'*26} | {'-'*4} | {'-'*22} | {'-'*24} | {'-'*7}")
    for t_range, d in benchmark_data["by_tempo"].items():
        _row(t_range, d)

    print("\n-- Breakdown by Chord Progression (3 Progressions) --")
    print(f" {'Progression':<26} | {'Ex':>4} | {'Pitch-Only (P / R / F1)':<22} | {'Pitch+String (P / R / F1)':<24} | {'F1 Gap':<7}")
    print(f" {'-'*26} | {'-'*4} | {'-'*22} | {'-'*24} | {'-'*7}")
    for pr, d in benchmark_data["by_progression"].items():
        _row(pr, d)

    print("=" * 102)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GuitarSet Comping Benchmark Evaluator")
    parser.add_argument("--dry-run", action="store_true", help="Run dry run on single excerpt")
    parser.add_argument("--jams", type=str, default="datasets/guitar/annotation/00_BN1-129-Eb_comp.jams", help="JAMS path for dry run")
    parser.add_argument("--audio", type=str, default="datasets/guitar/audio_mono-pickup_mix/00_BN1-129-Eb_comp_mix.wav", help="Audio path for dry run")
    parser.add_argument("--audio-type", choices=["mix", "mic"], default="mix", help="Audio type: mix or mic")
    parser.add_argument("--limit", type=int, default=18, help="Limit number of excerpts to evaluate (default: 18 stratified)")
    parser.add_argument("--tolerance", type=float, default=0.050, help="Onset matching tolerance in seconds (default: 0.050s)")
    parser.add_argument("--output-json", type=str, default="", help="Path to save evaluation JSON results")
    args = parser.parse_args()

    if args.dry_run:
        run_dry_run_on_excerpt(args.jams, args.audio, onset_tol=args.tolerance)
    else:
        suite = run_benchmark_suite(
            dataset_dir="datasets/guitar",
            audio_type=args.audio_type,
            limit=args.limit,
            onset_tol=args.tolerance
        )
        print_summary_report(suite)
        if args.output_json:
            os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
            with open(args.output_json, "w", encoding="utf-8") as f:
                json.dump(suite, f, indent=2)
            print(f"Benchmark results written to: {args.output_json}")
