import argparse
import sys
import os

# Ensure the prototype directory is in sys.path so local imports work when installed as a package
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

def main():
    parser = argparse.ArgumentParser(
        description="TabsGeneration CLI: Differentiable Ukulele Tab Transcription Pipeline",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")
    subparsers.required = True

    # --- TRANSCRIBE COMMAND ---
    transcribe_parser = subparsers.add_parser(
        "transcribe", 
        help="Transcribe a ukulele WAV file to tablature"
    )
    transcribe_parser.add_argument(
        "audio_file", 
        type=str, 
        help="Path to the ukulele WAV file to transcribe"
    )
    transcribe_parser.add_argument(
        "--notes", 
        type=int, 
        default=4, 
        choices=[1, 2, 3, 4], 
        help="Number of simulated voices/strings to extract (default: 4).\nUse 1 for mono pYIN path, 2-4 for polyphonic chord search."
    )
    transcribe_parser.add_argument(
        "--strummed", 
        type=str, 
        default="1,2,3,4", 
        help="Comma-separated list of strummed strings, e.g. '1,2,3' (default: 1,2,3,4)"
    )
    transcribe_parser.add_argument(
        "--iter", 
        type=int, 
        default=50, 
        help="Number of Adam optimization iterations (default: 50)"
    )
    transcribe_parser.add_argument(
        "--lr", 
        type=float, 
        default=0.5, 
        help="Learning rate for Adam optimizer (default: 0.5)"
    )

    # --- SYNTHESIZE COMMAND ---
    synth_parser = subparsers.add_parser(
        "synthesize", 
        help="Generate synthetic ukulele audio files using the Karplus-Strong model"
    )
    synth_parser.add_argument(
        "--out", 
        type=str, 
        default="synthetic_output.wav", 
        help="Output WAV file name/path (default: synthetic_output.wav)"
    )
    synth_parser.add_argument(
        "--frets", 
        type=str, 
        help="Comma-separated string:fret pairs to play, e.g. '4:0,3:0,2:0,1:3' for C-Major.\nString 1=A, 2=E, 3=C, 4=G."
    )
    synth_parser.add_argument(
        "--pitches", 
        type=str, 
        help="Comma-separated frequencies (Hz) to synthesize directly, e.g. '392.00,261.63'."
    )
    
    args = parser.parse_args()

    # Route to appropriate logic
    if args.command == "transcribe":
        from main import process_audio_multi_notes
        if not os.path.exists(args.audio_file):
            print(f"Error: Audio file '{args.audio_file}' not found.")
            sys.exit(1)
            
        strummed_strings = [int(s.strip()) for s in args.strummed.split(",") if s.strip().isdigit()]
        
        try:
            process_audio_multi_notes(
                file_path=args.audio_file, 
                num_notes=args.notes, 
                strummed_strings=strummed_strings, 
                num_iterations=args.iter, 
                lr=args.lr
            )
        except Exception as e:
            print(f"Transcription failed: {e}")
            sys.exit(1)

    elif args.command == "synthesize":
        from test_synthetic import generate_multi_note_audio, string_fret_to_hz
        
        pitches = []
        if args.frets:
            # Parse string:fret format
            pairs = args.frets.split(",")
            for pair in pairs:
                try:
                    s, f = map(int, pair.split(":"))
                    pitches.append(string_fret_to_hz(s, f))
                except ValueError:
                    print("Error: --frets must be formatted as 'string:fret,string:fret' (e.g. '1:3,2:0')")
                    sys.exit(1)
        elif args.pitches:
            try:
                pitches = [float(p.strip()) for p in args.pitches.split(",")]
            except ValueError:
                print("Error: --pitches must be a comma-separated list of numbers (e.g. '440.0,329.63')")
                sys.exit(1)
        else:
            print("Error: You must provide either --frets or --pitches to synthesize audio.")
            sys.exit(1)
            
        generate_multi_note_audio(pitches, args.out)
        print(f"Successfully generated {args.out}")

if __name__ == "__main__":
    main()
