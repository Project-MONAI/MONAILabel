# monailabel-client

Python client and CLI for scripts and automation. Included with MONAI Label.

Connect to a running server at `http://127.0.0.1:8000`:

```bash
monailabel-client login --username YOUR_USERNAME
monailabel-client projects
monailabel-client --help
```

Login prompts for the password. For another server, put `--url https://your-server` before the command. From source, prefix commands with `uv run`.

Python scripts use `Client` from `monailabel.client.client` with the saved login or `MONAILABEL_TOKEN`. See the server's `/docs` for API endpoints.
