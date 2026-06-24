# Docker Check

Docker deployment files are included:

```text
Dockerfile
docker-compose.yml
.dockerignore
```

Local verification command attempted:

```powershell
docker --version
```

Result in this environment:

```text
docker: The term 'docker' is not recognized as a name of a cmdlet, function,
script file, or executable program.
```

Conclusion: Docker proof could not be completed on this machine because Docker
is not installed or not available on `PATH`. The project is ready for Docker
testing on a machine with Docker Desktop or Docker Engine installed:

```powershell
docker compose up --build
```
