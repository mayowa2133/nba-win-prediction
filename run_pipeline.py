#!/usr/bin/env python
"""
Wrapper script to run the full pipeline from the project root.

This script simply calls the main pipeline script in src/pipeline/.
You can run it from the project root:

    python run_pipeline.py

Or directly:

    python src/pipeline/run_full_slate_pipeline.py
"""

import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    # Get the project root (where this script lives)
    project_root = Path(__file__).resolve().parent
    pipeline_script = project_root / "src" / "pipeline" / "run_full_slate_pipeline.py"
    
    if not pipeline_script.exists():
        print(f"ERROR: Pipeline script not found at {pipeline_script}")
        sys.exit(1)
    
    # Run the pipeline script with all arguments passed through
    sys.exit(subprocess.run([sys.executable, str(pipeline_script)] + sys.argv[1:]).returncode)

