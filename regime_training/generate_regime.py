"""
Regime-Conditional Synthetic Rainfall Generator (Dispatcher)
============================================================
Dispatches execution to either:
  - generate_regime_v5.py (Transition-Aware Markovian Block Assembly + OLA, default)
  - generate_regime_v4.py (Legacy Random Permutation Shuffling)

Usage
-----
  python regime_training/generate_regime.py --version v5 [ARGS...]
  python regime_training/generate_regime.py --version v4 [ARGS...]
"""

import sys
import os

def main():
    args = sys.argv[1:]
    version = "v5"
    
    # Check if --version flag is provided
    if "--version" in args:
        idx = args.index("--version")
        if idx + 1 < len(args):
            version = args[idx + 1]
            # Remove --version and its value from sys.argv for delegated script
            del sys.argv[idx:idx+2]
            
    if version.lower() == "v4":
        from regime_training.generate_regime_v4 import main as v4_main
        v4_main()
    else:
        from regime_training.generate_regime_v5 import main as v5_main
        v5_main()

if __name__ == "__main__":
    main()
