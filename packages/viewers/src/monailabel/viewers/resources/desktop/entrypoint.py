"""Run one native viewer on a private X/VNC display inside its container."""

import json
import os
import secrets
import subprocess
import time
from pathlib import Path


def main() -> None:
    os.umask(0o077)
    session = Path("/session")
    (session / "tmp").mkdir(exist_ok=True)
    Path("/tmp/.X11-unix").mkdir(mode=0o1777, exist_ok=True)
    settings = json.loads((session / "runtime.json").read_text())
    # Host networking lets the bridge reach a backend bound to localhost. X requires
    # a private cookie; VNC has no TCP listener and uses a private Unix socket only.
    os.environ["DISPLAY"] = ":" + str(1000 + secrets.randbelow(59000))
    os.environ["XAUTHORITY"] = str(session / ".Xauthority")
    subprocess.run(["xauth", "add", os.environ["DISPLAY"], ".", secrets.token_hex(16)], check=True)
    vnc = subprocess.Popen(
        [
            "Xtigervnc",
            os.environ["DISPLAY"],
            "-geometry",
            "1600x1000",
            "-depth",
            "24",
            "-nolisten",
            "tcp",
            "-auth",
            os.environ["XAUTHORITY"],
            "-rfbport",
            "-1",
            "-rfbunixpath",
            "/connection/rfb.sock",
            "-rfbunixmode",
            "0600",
            "-SecurityTypes",
            "None",
            "-AlwaysShared",
            "-desktop",
            "MONAI Label " + settings["viewer"],
        ]
    )
    try:
        for _ in range(100):
            if vnc.poll() is not None:
                raise RuntimeError("The private display could not start.")
            ready = subprocess.run(
                ["xdpyinfo"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            if ready.returncode == 0:
                break
            time.sleep(0.1)
        else:
            raise RuntimeError("The private display did not become ready.")
        subprocess.Popen(["openbox", "--config-file", "/etc/monailabel-openbox.xml"])
        executable = settings["executable"]
        if settings["viewer"] == "qupath":
            # Build and install using QuPath's own compiler and private preferences.
            source = Path("/bridges/qupath")
            classes = session / "classes"
            classes.mkdir(exist_ok=True)
            subprocess.run(
                [
                    executable,
                    "script",
                    str(source / "build.groovy"),
                    "--args",
                    str(source),
                    "--args",
                    str(classes),
                ],
                check=True,
            )
            import zipfile

            jar = session / "monailabel-qupath.jar"
            with zipfile.ZipFile(jar, "w", zipfile.ZIP_DEFLATED) as archive:
                for path in classes.rglob("*.class"):
                    archive.write(path, str(path.relative_to(classes)))
                archive.writestr(
                    "META-INF/services/qupath.lib.gui.extensions.QuPathExtension",
                    "org.monailabel.qupath.MonaiLabelExtension\n",
                )
            subprocess.run(
                [
                    executable,
                    "script",
                    str(source / "install.groovy"),
                    "--args",
                    str(jar),
                    "--args",
                    str(session / "qupath"),
                ],
                check=True,
            )
            command = [executable, "--quiet"]
        else:
            command = [executable, "--no-splash", "--python-script", "/bridges/slicer_bridge.py"]
        viewer = subprocess.Popen(command)
        time.sleep(2)
        if viewer.poll() is not None:
            raise RuntimeError("The native viewer exited during startup.")
        (session / "started").touch()
        while viewer.poll() is None:
            if vnc.poll() is not None:
                viewer.terminate()
                raise RuntimeError("The private display stopped.")
            time.sleep(1)
        if viewer.returncode:
            raise RuntimeError("The native viewer stopped unexpectedly.")
    finally:
        vnc.terminate()


if __name__ == "__main__":
    main()
