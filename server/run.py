"""
Launch the Streamlit dashboard.

    python server/run.py

The dashboard runs its own scanner thread, so there is no reason to start
main.py alongside it — the previous version launched both, which duplicated
every scan and doubled the load on Yahoo. Use main.py on its own for a
terminal-only scan.
"""
import subprocess
import sys
from pathlib import Path

UI = Path(__file__).parent / 'ui.py'


def run_app():
    # Absolute path: the old relative "app.py" only resolved when the cwd
    # happened to be server/.
    return subprocess.run([sys.executable, '-m', 'streamlit', 'run', str(UI)]).returncode


if __name__ == '__main__':
    sys.exit(run_app())
