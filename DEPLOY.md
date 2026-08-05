# Deploying the monitor station (Oracle Cloud Always Free)

This walks through putting `server/web/app.py` on Oracle Cloud's Always Free
tier so other people can reach it, with HTTP Basic Auth turned on so it isn't
open to the internet.

Console menu wording drifts as Oracle updates the UI — treat step names as
"look for something like this," not a pixel-exact script.

## 0. What you're deploying

A single Docker container running the FastAPI + SSE monitor station
(`server/run.py`). It's one process with in-memory scanner state — no
database, no external services required. The `Dockerfile` at the repo root
builds it; see its header comment for the full `docker run` command.

Two things worth knowing going in:
- `server/rl/checkpoints/best.pt` is gitignored — a fresh clone has no RL
  checkpoint, so the RL blending in the dashboard stays inactive (soft-fails,
  same as local dev with no checkpoint) until you copy one over or train on
  the server.
- Every core dependency (numpy, pandas, polars, torch, fastapi, etc.) has a
  published ARM64 (aarch64) Linux wheel, checked against PyPI directly —
  `pip install -r requirements.txt` should resolve cleanly on Oracle's ARM
  shape without needing a Rust/C toolchain for source builds.

## 1. Create the free VM

1. Sign up / log into [Oracle Cloud](https://cloud.oracle.com) (Always Free
   sign-up asks for a card for identity verification but does not charge for
   Always Free resources — watch for accidentally provisioning something
   outside the Always Free shapes).
2. **Compute → Instances → Create Instance.**
3. Image: **Ubuntu 22.04** (or newer).
4. Shape: **Ampere A1 (VM.Standard.A1.Flex)**, the Always Free ARM shape —
   you get up to 4 OCPUs / 24GB RAM total across your Always Free A1
   instances, which is generous for this app.
5. Add your SSH public key (or let Oracle generate a key pair and download
   it) — you'll need it to log in.
6. Create the instance and note its **public IP address**.

## 2. Open the port

Two layers both default to blocking inbound traffic — you need both open, and
forgetting the second one is the most common way this "doesn't work":

1. **OCI Security List / Network Security Group**: on the instance's VCN
   subnet, add an ingress rule allowing TCP on whatever port you'll expose
   (8000 if going straight to the app, or 80/443 if fronting with a reverse
   proxy — see step 5).
2. **The VM's own firewall**: Oracle's Ubuntu images ship with `iptables`
   rules that block everything but SSH by default, independent of the OCI
   console setting above. SSH in and run:
   ```bash
   sudo iptables -I INPUT -p tcp --dport 8000 -j ACCEPT
   sudo netfilter-persistent save   # or: sudo apt install iptables-persistent
   ```

## 3. Install Docker

```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER
# log out and back in for the group change to apply
```

## 4. Get the code onto the VM

```bash
git clone <your-fork-or-repo-url> smartraid
cd smartraid
```

(No public repo? `scp -r` the project directory instead — just exclude `env/`
and `.git/`, which `.dockerignore` already keeps out of the *image* but you
don't need to copy them to the VM at all.)

## 5. Build and run

```bash
docker build -t smartraid .

docker run -d --name smartraid --restart unless-stopped \
  -p 8000:8000 \
  -e SMARTRAID_USER=youruser \
  -e SMARTRAID_PASSWORD='something-long-and-random' \
  -v smartraid-checkpoints:/app/server/rl/checkpoints \
  smartraid
```

`--restart unless-stopped` brings it back after a VM reboot (Docker's own
systemd service is enabled by the install script in step 3, so the daemon
itself comes back too).

Visit `http://<the-VM's-public-IP>:8000` — your browser will prompt for the
username/password you set above.

**Set `SMARTRAID_USER`/`SMARTRAID_PASSWORD` — this is not optional.**
Without them the app has no authentication at all (see `server/run.py`'s
warning), and you're about to bind it to a public IP.

## 6. Optional but recommended: HTTPS

Plain HTTP Basic Auth sends the password base64-encoded, not encrypted —
trivial to read off the wire. Fine on a network you trust; not fine over the
open internet. Two free ways to fix it, pick one:

- **Cloudflare Tunnel** (`cloudflared`) running on the same VM: gives you a
  real HTTPS URL with no inbound port-opening at all — you can skip step 2
  entirely if you go this route. Simplest option if you don't already own a
  domain.
- **Caddy** as a reverse proxy in front of port 8000, if you do have a domain
  pointed at the VM's IP: `caddy reverse-proxy --from yourdomain.com --to
  localhost:8000` gets you automatic Let's Encrypt HTTPS in one line.

## 7. Updating later

```bash
cd smartraid
git pull
docker build -t smartraid .
docker stop smartraid && docker rm smartraid
docker run -d --name smartraid --restart unless-stopped -p 8000:8000 \
  -e SMARTRAID_USER=youruser -e SMARTRAID_PASSWORD='...' \
  -v smartraid-checkpoints:/app/server/rl/checkpoints \
  smartraid
```
