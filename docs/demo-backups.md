# Demo backups and restore rehearsal

`scripts/demo_backup.py` makes a verified cold copy of an existing workspace using Python 3.12 or newer. It never stops services, requests administrator access, deletes an old backup or overwrites a restore destination. Prepare and test the procedure with disposable data before an approved maintenance window. Do not run it as root.

## Before copying

Ask visitors to submit or save their annotations and native viewer drafts, then stop all writers to this workspace: the backend, inference/training jobs, browser desktops and any externally managed viewer services. The script acquires the backend's existing `server.lock` using the same Linux file-lock mechanism as the application. It refuses an active backend, but that lock cannot establish that an independently running viewer or external script has stopped. `--writers-stopped` is an operator acknowledgement, not a command that stops them.

Choose an existing backup directory owned by the backup user with mode `0700`. Prefer storage on another device. A new uniquely named subdirectory is created for each backup, never merged into a previous copy. Check capacity before selecting a retention policy: copies are full, uncompressed, and not deduplicated. The script leaves at least 1 GiB of free space at its initial capacity check; this is not a reservation against other processes writing to the disk.

The copied workspace includes its database and any WAL files, `secrets.key`, annotations, artifacts, workspace-contained viewer drafts, quota ledgers and audit state. It excludes the top-level `.cache` directory and the transient `server.lock`. It refuses links, separate mounted filesystems and special files, including sockets, rather than silently omit user data or follow a path outside the reviewed workspace. Inspect any such refusal; do not remove drafts or data to make a backup succeed. Stale runtime sockets need a separately reviewed cleanup after all owners have stopped.

This is a workspace backup, not a full-machine backup. Separately preserve the selected source/environment versions, service and proxy configuration, certificate keys, firewall configuration and any external viewer volumes. Downloaded public weights/viewers may be reproducible, but retain the exact version/checksum information and any custom trained weights. Keep credentials in protected backup storage, never in source control or chat. The files are access-restricted, not encrypted by this script; use an encrypted destination when physical loss or other disk users are in scope.

## Create and verify

Replace the example paths with the reviewed locations. First inspect a no-write plan while the workspace is stopped:

```bash
uv run python scripts/demo_backup.py create --workspace /path/to/workspace --destination /path/to/private-backups
```

After confirming all writers are stopped:

```bash
uv run python scripts/demo_backup.py create --workspace /path/to/workspace --destination /path/to/private-backups --writers-stopped --apply
```

The result identifies the created `backup-...` directory. Every regular file is copied, flushed and verified against a SHA-256 manifest before that name is published. A failed attempt retains an `.incomplete-...` directory for investigation; it must not be treated as a completed backup. The original workspace and older backups are not removed. Independently re-verify the copy after transferring it to the backup device:

```bash
uv run python scripts/demo_backup.py verify /path/to/private-backups/backup-example
```

## Restore rehearsal

Restore only into a new, nonexistent directory under an existing private directory:

```bash
uv run python scripts/demo_backup.py restore-copy /path/to/private-backups/backup-example --target /path/to/private-restore/rehearsal
```

Checksums are verified before creating the target and during copying. The script does not start an application or replace the original workspace. In the isolated copy, check database integrity, credentials, representative saved annotations and viewer drafts using the compatible application version. Do not enable outbound integrations or expose a restored test copy publicly. External viewer volumes and configurations must also be restored into separate locations before testing their workflows.

For a daily backup during a short pilot, agree on a quiet maintenance window and a way to save/end sessions first. Do not attach this cold-copy command to an unattended service-stop timer that could interrupt unsaved work. Automatic retention/deletion and service stopping are intentionally not implemented by this helper.

## Cutover and rollback

After the final backup is verified, switch the reviewed source/environment configuration while preserving the workspace and enforcing one backend owner. If the new version fails, first close the new public entry point. Revert software only when its database compatibility is established; otherwise preserve the newer workspace for diagnosis and request a recovery decision. Never copy an old backup over newer annotations merely to recover service availability. See [HTTPS deployment](https-deployment.md) for the separate public-network approval gate.
