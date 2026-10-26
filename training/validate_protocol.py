"""
ASVspoof Protocol and Dataset Validation Tool.
Validates official ASVspoof protocol files against audio directories,
reporting missing files, label distributions, and class counts without fabrication.
"""
import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def validate_asvspoof_protocol(
    protocol_path: str,
    audio_dir: str,
    max_missing_examples: int = 5
) -> Dict:
    """
    Validate an ASVspoof protocol file and check corresponding audio files.

    Args:
        protocol_path: Path to ASVspoof protocol file (e.g. ASVspoof2019.LA.cm.train.trn.txt)
        audio_dir: Directory containing the audio files (e.g. flac/)
        max_missing_examples: Maximum number of missing files to list in detail

    Returns:
        Dictionary with validation results
    """
    proto_file = Path(protocol_path)
    audio_path = Path(audio_dir)

    if not proto_file.is_file():
        return {
            "valid": False,
            "error": f"Protocol file does not exist: {protocol_path}",
            "protocol_path": str(proto_file),
            "audio_dir": str(audio_path)
        }

    if not audio_path.is_dir():
        return {
            "valid": False,
            "error": f"Audio directory does not exist: {audio_dir}",
            "protocol_path": str(proto_file),
            "audio_dir": str(audio_path)
        }

    total_lines = 0
    malformed_lines = []
    bonafide_count = 0
    spoof_count = 0
    unknown_labels = []
    found_files = 0
    missing_files = []

    # Valid extensions in ASVspoof corpora
    valid_exts = (".flac", ".wav", ".mp3", ".m4a")

    with proto_file.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                continue

            total_lines += 1
            fields = line_str.split()
            if len(fields) < 2:
                malformed_lines.append((line_num, line_str))
                continue

            # In ASVspoof protocols, utterance ID is typically field 1:
            # Format: SPEAKER_ID UTTERANCE_ID ... LABEL (last field)
            utterance_id = fields[1]
            raw_label = fields[-1].lower().replace("_", "").replace("-", "")

            if raw_label in {"bonafide", "genuine", "real"}:
                bonafide_count += 1
            elif raw_label in {"spoof", "fake"}:
                spoof_count += 1
            else:
                unknown_labels.append((line_num, utterance_id, fields[-1]))

            # Check if audio file exists
            candidates = [audio_path / utterance_id]
            if not Path(utterance_id).suffix:
                candidates.extend(audio_path / f"{utterance_id}{ext}" for ext in valid_exts)

            if any(c.is_file() for c in candidates):
                found_files += 1
            else:
                missing_files.append(utterance_id)

    missing_count = len(missing_files)
    is_valid = (
        total_lines > 0
        and len(malformed_lines) == 0
        and len(unknown_labels) == 0
        and missing_count == 0
        and bonafide_count > 0
        and spoof_count > 0
    )

    return {
        "valid": is_valid,
        "protocol_path": str(proto_file.resolve()),
        "audio_dir": str(audio_path.resolve()),
        "total_protocol_entries": total_lines,
        "malformed_entries_count": len(malformed_lines),
        "malformed_examples": malformed_lines[:max_missing_examples],
        "class_counts": {
            "bonafide": bonafide_count,
            "spoof": spoof_count,
            "unknown": len(unknown_labels)
        },
        "found_audio_files": found_files,
        "missing_audio_files_count": missing_count,
        "missing_examples": missing_files[:max_missing_examples],
        "both_classes_present": bool(bonafide_count > 0 and spoof_count > 0),
        "summary": (
            f"Validated {total_lines} entries: {bonafide_count} bona-fide, {spoof_count} spoof. "
            f"Found {found_files} files, {missing_count} missing."
        )
    }


def main():
    parser = argparse.ArgumentParser(
        description="Validate ASVspoof protocol files and verify audio dataset completeness."
    )
    parser.add_argument("--protocol", required=True, type=str, help="Path to ASVspoof protocol text file")
    parser.add_argument("--audio-dir", "--audio_dir", dest="audio_dir", required=True, type=str, help="Directory containing audio files")
    parser.add_argument("--max-missing", "--max_missing", dest="max_missing", type=int, default=5, help="Number of missing file examples to display")

    args = parser.parse_args()

    results = validate_asvspoof_protocol(
        protocol_path=args.protocol,
        audio_dir=args.audio_dir,
        max_missing_examples=args.max_missing
    )

    print("\n" + "=" * 60)
    print("ASVSPOOF PROTOCOL & DATASET VALIDATION REPORT")
    print("=" * 60)
    print(f"Protocol File    : {results.get('protocol_path')}")
    print(f"Audio Directory  : {results.get('audio_dir')}")
    print(f"Total Entries    : {results.get('total_protocol_entries', 0)}")
    print(f"Classes Found    : {results.get('class_counts', {})}")
    print(f"Audio Files Found: {results.get('found_audio_files', 0)}")
    print(f"Missing Files    : {results.get('missing_audio_files_count', 0)}")

    if results.get("missing_examples"):
        print(f"Missing Examples : {', '.join(results['missing_examples'])}")

    if results.get("error"):
        print(f"\n[ERROR] {results['error']}")
        sys.exit(1)

    if results["valid"]:
        print("\n[SUCCESS] Dataset strictly matches protocol specification. Ready for evaluation/training.")
        sys.exit(0)
    else:
        print("\n[INCOMPLETE] Dataset has missing audio files or class imbalances.")
        sys.exit(1)


if __name__ == "__main__":
    main()
