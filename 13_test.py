#!/usr/bin/env python3
"""
Simple simulator for the Foxy jumpscare random roll logic.

Matches mod behavior:
- one random roll every `check_interval_seconds`
- trigger chance of `chance_percent` per roll
"""

from __future__ import annotations

import argparse
import os
import random
import statistics
import sys
import time


def simulate_session(chance_percent: float, check_interval_seconds: int, session_seconds: int) -> int:
    chance = max(0.0, min(100.0, chance_percent)) / 100.0
    interval = max(1, check_interval_seconds)
    checks = session_seconds // interval
    hits = 0

    for _ in range(checks):
        if random.random() < chance:
            hits += 1

    return hits


def play_beep_jumpscare() -> None:
    """Play a loud-ish beep stack on Windows as a local test fallback."""
    if sys.platform != "win32":
        print("Audio test beep is only supported on Windows in this script.")
        return

    try:
        import winsound  # type: ignore
    except Exception as exc:  # pragma: no cover
        print(f"Could not import winsound: {exc}")
        return

    pattern = [
        (1400, 220),
        (900, 240),
        (1700, 260),
    ]
    for frequency, duration_ms in pattern:
        winsound.Beep(frequency, duration_ms)
        time.sleep(0.03)


def play_wav_jumpscare(wav_path: str) -> bool:
    """Play a WAV file on Windows. Returns True on success."""
    if sys.platform != "win32":
        print("WAV playback in this script is only supported on Windows.")
        return False

    wav_path = os.path.abspath(wav_path)
    if not os.path.isfile(wav_path):
        print(f"WAV file not found: {wav_path}")
        return False

    try:
        import winsound  # type: ignore
    except Exception as exc:  # pragma: no cover
        print(f"Could not import winsound: {exc}")
        return False

    try:
        flags = winsound.SND_FILENAME
        # Keep playback blocking so the process does not exit before audio is heard.
        if hasattr(winsound, "SND_SYNC"):
            flags |= winsound.SND_SYNC
        winsound.PlaySound(wav_path, flags)
        return True
    except Exception as exc:  # pragma: no cover
        print(f"Could not play WAV file: {exc}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate Foxy jumpscare trigger frequency.")
    parser.add_argument("--chance", type=float, default=4.0, help="Chance per check in percent (default: 4)")
    parser.add_argument(
        "--interval",
        type=int,
        default=25,
        help="Check interval in seconds (default: 25)",
    )
    parser.add_argument(
        "--minutes",
        type=int,
        default=30,
        help="Session length in minutes per trial (default: 30)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=1000,
        help="Number of sessions to simulate (default: 1000)",
    )
    parser.add_argument("--seed", type=int, default=1337, help="Random seed (default: 1337)")
    parser.add_argument(
        "--play",
        action="store_true",
        help="Play local jumpscare audio once for each simulated hit in the first trial.",
    )
    parser.add_argument(
        "--wav",
        type=str,
        default=os.path.join("FoxyJumpscareMod", "Assets", "Sounds", "foxy_jumpscare.wav"),
        help="Path to WAV file used by --play (default: mod jumpscare WAV)",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    session_seconds = max(1, args.minutes) * 60
    results = [
        simulate_session(args.chance, args.interval, session_seconds)
        for _ in range(max(1, args.trials))
    ]

    checks_per_session = session_seconds // max(1, args.interval)
    expected = checks_per_session * (max(0.0, min(100.0, args.chance)) / 100.0)

    print("Foxy Jumpscare Simulator")
    print("------------------------")
    print(f"chance per check: {args.chance:.3f}%")
    print(f"check interval:   {args.interval}s")
    print(f"session length:   {args.minutes} minute(s)")
    print(f"trials:           {args.trials}")
    print()
    print(f"checks/session:   {checks_per_session}")
    print(f"expected hits:    {expected:.3f}")
    print(f"mean hits:        {statistics.mean(results):.3f}")
    print(f"median hits:      {statistics.median(results):.3f}")
    print(f"min/max hits:     {min(results)} / {max(results)}")

    if args.play:
        first_trial_hits = simulate_session(args.chance, args.interval, session_seconds)
        play_count = max(1, first_trial_hits)
        print()
        if first_trial_hits == 0:
            print("simulated 0 hits, forcing 1 preview playback...")
        else:
            print(f"playing audio test for {first_trial_hits} simulated jumpscare(s)...")
        print(f"audio source: {os.path.abspath(args.wav)}")
        for idx in range(play_count):
            print(f"playback {idx + 1}/{play_count}")
            if not play_wav_jumpscare(args.wav):
                play_beep_jumpscare()


if __name__ == "__main__":
    main()
