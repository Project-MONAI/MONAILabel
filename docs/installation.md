# Install MONAI Label

Use Linux and Python 3.12 or 3.13. Install the [system prerequisites](../README.md#system-requirements) for the viewers and models you need. When upgrading from 0.x, use a fresh environment; legacy apps and workspaces are not migrated automatically.

## Python package

Choose **Miniconda** or **venv** to create an environment, then install MONAI Label.

### Miniconda

[Miniconda](https://repo.anaconda.com/miniconda/) installs Python and conda. These Linux commands select the installer for your architecture:

```bash
curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-$(uname -m).sh -o miniconda.sh
bash miniconda.sh -b -p "$HOME/miniconda3"
source "$HOME/miniconda3/bin/activate"

conda create -n monailabel -c conda-forge python=3.12 pip -y
conda activate monailabel
```

### venv

[venv](https://docs.python.org/3/library/venv.html) is Python's environment tool. On Ubuntu 24.04, install Python and venv, then activate an environment:

```bash
sudo apt-get update && sudo apt-get install -y python3.12-venv
python3.12 -m venv .venv
source .venv/bin/activate
```

### Install and run

After the release candidate is published to PyPI:

```bash
pip install monailabel -U --pre
monailabel
```

Open **http://localhost:8000** and create the administrator. Data is saved in `./workspace/`. See [assistant setup](coordinator.md#setup) for chat configuration.

Use the same install command to upgrade. `--pre` includes release candidates; for stable releases, use `pip install monailabel -U`. Dependencies are installed automatically on Linux x86_64 and ARM64. In a new terminal, activate your environment before running `monailabel`.

## From source

Follow the [README Quickstart](../README.md#quickstart), then start with `uv run monailabel`. See [build and release](releasing.md) for local wheels and installation checks.

## Accounts and project roles

Administrators use **Team & roles → Create user**, then **Assign roles** to grant manager, annotator or reviewer access to a project. A new account has no project access until roles are assigned.

API equivalents: `POST /api/auth/users` and `PUT /api/projects/{project_id}/members`, with `user_id` and `roles` in the membership body.

See [Docker setup](docker.md) and [third-party licenses](../THIRD_PARTY_NOTICES.md).
