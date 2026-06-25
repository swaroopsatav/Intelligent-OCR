# Docker Check

Docker deployment files are included and syntax-checked:

```text
Dockerfile
docker-compose.yml
.dockerignore
```

Local verification commands:

```powershell
.\venv\Scripts\python scripts\verify_docker.py
docker compose build
```

Current result:

- Docker CLI is installed: Docker `29.5.3`.
- `docker compose config` succeeds, so the Compose file is valid.
- `docker compose build` completed successfully on this machine and produced
  the backend and frontend images.

The machine-readable verification result is stored in
`docs/docker-check.json`.

To run the built services, start Docker Desktop or Docker Engine, then run:

```powershell
docker compose up
```
