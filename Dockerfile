# SmarTraid monitor station (FastAPI + SSE) — server/web/app.py.
#
# Build (from the repo root):
#   docker build -t smartraid .
#
# Run:
#   docker run -d --name smartraid --restart unless-stopped -p 8000:8000 \
#     -e SMARTRAID_USER=youruser -e SMARTRAID_PASSWORD=yourpassword \
#     -v smartraid-checkpoints:/app/server/rl/checkpoints \
#     smartraid
#
# SMARTRAID_USER / SMARTRAID_PASSWORD turn on HTTP Basic Auth for the whole
# app (server/web/auth.py) — set both before exposing this beyond localhost;
# see server/run.py's warning. Leaving them unset matches local dev (no auth).
#
# The checkpoints volume is optional: server/rl/checkpoints/best.pt is
# gitignored (not baked into the image), so RL blending stays inactive until
# either you copy a trained checkpoint in or train one on the container itself.

FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

COPY server/ ./server/

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

CMD ["python", "server/run.py", "--host", "0.0.0.0", "--port", "8000"]
