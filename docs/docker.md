# Run with Docker

Follow the [system requirements](../README.md#system-requirements). Run these commands from the repository folder.

The same commands work on Linux x86_64, DGX Spark and Jetson Thor (ARM64); Docker selects the host architecture.

## Start

```bash
docker compose up -d --build
```

Open **http://localhost:8000** and create the administrator. Data and downloads are saved in `./workspace/`, the same default as a source installation. The [Compose file](../compose.yaml) configures the mounts and networking needed by managed services and viewers.

## Everyday commands

```bash
# Start again
docker compose up -d

# Stop
docker compose stop

# Restart
docker compose restart

# View logs
docker compose logs -f
```

After updating the source, run `docker compose up -d --build` again. Stopping or rebuilding the container keeps your workspace.

## Deployment settings

Defaults work without configuration. To change the workspace or port, put these optional settings in `.env` in the repository folder, then run `docker compose up -d`:

```dotenv
MONAILABEL_WORKSPACE_DIR=/absolute/path/to/workspace
MONAILABEL_PORT=8000
```

For GPT Astra, export `OPENAI_API_KEY` in your shell and add these settings to `.env`:

```dotenv
MONAILABEL_ASSISTANT_PROVIDER=openai
MONAILABEL_ASSISTANT_MODEL=gpt-6-astra
```

Run `docker compose up -d` to apply environment changes. The same key enables direct Astra annotation. See [assistant setup](coordinator.md#setup) and [remote access](viewers.md#phones-tablets-and-other-computers).

For HTTPS, append `"--https"` to the service's `command` in `compose.yaml`, then run `docker compose up -d`. Follow the [certificate trust step](viewers.md#https) on each browser device for voice input.

- Run one server per workspace. Back up the whole workspace, including `secrets.key`.
- Managed services use the host Docker socket. Run this deployment only on a trusted host.
- Workspace files are owned by the container's root user.

## Verify a build

```bash
uv run python scripts/check_docker.py monailabel:local
```

This checks installed packages, server startup and workspace persistence in disposable containers. License notices and the dependency inventory are under `/usr/share/monailabel/` inside the image.
