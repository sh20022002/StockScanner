"""
Launch the monitor station (FastAPI + Server-Sent Events).

    python server/run.py                  # http://127.0.0.1:8000
    python server/run.py --port 8080
    python server/run.py --reload         # dev: reload on file change
    python server/run.py --host 0.0.0.0   # expose beyond localhost — see the
                                           # warning below before you do this

The scanner runs as a background task inside this same process and is
controlled from the page (Start/Stop) rather than from the command line — see
server/web/app.py. There is no reason to also run main.py alongside it: that
would duplicate every scan and double the load on Yahoo.
"""
import argparse
import multiprocessing
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description='Run the SmarTraid monitor station.')
    parser.add_argument('--host', default='127.0.0.1',
                        help='Bind address. The app has no authentication — '
                             'do not bind 0.0.0.0 outside a trusted network.')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--reload', action='store_true', help='Auto-reload on file changes (dev only).')
    args = parser.parse_args()

    import uvicorn
    uvicorn.run('web.app:app', host=args.host, port=args.port,
               reload=args.reload, app_dir=os.path.dirname(os.path.abspath(__file__)))


if __name__ == '__main__':
    # The scanner's process pool (scanner.py) uses spawn on Windows, which
    # re-imports this module in each worker — freeze_support is required so
    # that re-import doesn't try to launch a second server.
    multiprocessing.freeze_support()
    main()
