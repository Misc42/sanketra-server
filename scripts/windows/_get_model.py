#!/usr/bin/env python3
"""Helper script for run_bg.bat to get custom model selection"""
import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))
src_dir = os.path.join(root_dir, 'src')
sys.path.insert(0, src_dir)

try:
    from stt_common import custom_model_selection
    model, compute, device = custom_model_selection()

    # Write to temp file for batch to read
    with open(os.path.join(script_dir, '_model_choice.tmp'), 'w') as f:
        f.write(f"{model},{compute}")

    sys.exit(0)
except KeyboardInterrupt:
    sys.exit(1)
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
