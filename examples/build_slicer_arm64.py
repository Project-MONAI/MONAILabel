"""Build a portable native Slicer in an isolated Linux ARM64 Docker container.

uv run python examples/build_slicer_arm64.py

The first build can take hours and requires substantial disk space. Sources,
dependencies and intermediate objects are cached so interrupted builds resume.
No GPU, workspace database, credentials or existing viewer installations are used.
"""

import argparse
import hashlib
import os
import platform
import secrets
import shlex
import shutil
import subprocess
import tempfile
from contextlib import suppress
from pathlib import Path

import httpx
from filelock import FileLock

from monailabel.viewers.provisioning.desktop import extract

SLICER_REVISION = "4e21c19d8360242d88d6ca2490bf52400441b8e1"
DCMQI_REVISION = "a1022985d4338935cc9aebe1145bac1d8d896cd6"
WRAPPER_URL = (
    "https://files.pythonhosted.org/packages/fe/6b/"
    "415be891249b4133c4803eea049ce4546f9167d74877c7ed3d2b7d89635e/dcmqi-0.4.0.tar.gz"
)
WRAPPER_SHA256 = "d93c819e79801ca8c2b481b95f3a0023e45c3a82d568321bd3eac9abd17fb4d9"
BUILD = "/work/build-v5.12.4"


def run(*args):
    subprocess.run(args, check=True)


def write_changed(path, content):
    if not path.exists() or path.read_text() != content:
        path.write_text(content)


def show_installation(destination):
    executables = list(destination.glob("*/Slicer"))
    if len(executables) != 1 or not os.access(executables[0], os.X_OK):
        raise RuntimeError(f"Incomplete portable installation in {destination}; move it aside.")
    with executables[0].open("rb") as stream:
        header = stream.read(20)
    if header[:6] != b"\x7fELF\x02\x01" or header[18:20] != b"\xb7\x00":
        raise RuntimeError("The built Slicer launcher is not a native ARM64 ELF executable.")
    print(f"export MONAILABEL_SLICER_EXECUTABLE={shlex.quote(str(executables[0]))}")


def source(root, name, repository, revision):
    directory = root / name
    if not directory.exists():
        directory.mkdir()
        run("git", "-C", str(directory), "init")
        run("git", "-C", str(directory), "fetch", "--depth=1", repository, revision)
        run("git", "-C", str(directory), "checkout", "--detach", "FETCH_HEAD")
    actual = subprocess.check_output(
        ["git", "-C", str(directory), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != revision:
        raise RuntimeError(f"Unexpected source revision in {directory}; use a separate cache.")
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path("workspace/.cache/tools/slicer"))
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.machine() not in {"aarch64", "arm64"}:
        parser.error("This recipe requires a native Linux ARM64 Docker host.")
    if not shutil.which("docker") or not 1 <= args.jobs <= 16:
        parser.error("Docker and --jobs between 1 and 16 are required.")
    root = args.cache_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root / "native-build.lock"), timeout=1):
        destination = root / "native-install"
        if destination.exists():
            show_installation(destination)
            return
        slicer = source(
            root, "source-v5.12.4", "https://github.com/Slicer/Slicer.git", SLICER_REVISION
        )
        source(root, "dcmqi-v1.5.4", "https://github.com/QIICR/dcmqi.git", DCMQI_REVISION)
        # Upstream identifies every 64-bit Linux target as amd64, including
        # native ARM builds. Correct the package and application metadata.
        architecture = slicer / "CMake/SlicerMacroGetOperatingSystemArchitectureBitness.cmake"
        original = "    set(${MY_VAR_PREFIX}_ARCHITECTURE amd64)"
        replacement = (
            '    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")\n'
            "      set(${MY_VAR_PREFIX}_ARCHITECTURE arm64)\n"
            "    else()\n"
            "      set(${MY_VAR_PREFIX}_ARCHITECTURE amd64)\n"
            "    endif()"
        )
        content = architecture.read_text()
        if 'MATCHES "^(aarch64|arm64)$"' not in content:
            if original not in content:
                raise RuntimeError("The pinned Slicer architecture detection has changed.")
            write_changed(architecture, content.replace(original, replacement))
        # Upstream wheels contain x86 Linux executables. Install our native build
        # below instead, retaining the pinned Python package and all seven CLIs.
        requirements = slicer / "SuperBuild/External_python-dicom-requirements.cmake"
        content = requirements.read_text()
        write_changed(
            requirements,
            content.replace(
                "dcmqi==0.4.0 --hash=", "dcmqi==0.4.0; platform_machine != 'aarch64' --hash="
            ),
        )
        archive = root / "dcmqi-0.4.0.tar.gz"
        if not archive.exists():
            response = httpx.get(WRAPPER_URL, follow_redirects=True, timeout=60)
            response.raise_for_status()
            archive.write_bytes(response.content)
        if hashlib.sha256(archive.read_bytes()).hexdigest() != WRAPPER_SHA256:
            raise RuntimeError(f"Checksum mismatch: {archive}")
        if not (root / "dcmqi-0.4.0").exists():
            extract(archive, root, "tar.gz")
        write_changed(
            root / "dcmqi-0.4.0/CMakeLists.txt",
            "cmake_minimum_required(VERSION 3.15...3.26)\n"
            "project(${SKBUILD_PROJECT_NAME} LANGUAGES NONE)\n"
            'file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/binaries.txt" binaries)\n'
            "foreach(binary IN LISTS binaries)\n"
            f'  install(PROGRAMS "{BUILD}/Slicer-build/lib/Slicer-5.12/cli-modules/${{binary}}" '
            'DESTINATION "dcmqi/bin")\n'
            "endforeach()\n",
        )
        dockerfile = Path(__file__).with_name("slicer-arm64.Dockerfile")
        signature = hashlib.sha256(dockerfile.read_bytes()).hexdigest()[:12]
        image = f"monailabel-slicer-builder:5.12.4-arm64-{signature}"
        run(
            "docker",
            "build",
            "--platform",
            "linux/arm64",
            "-t",
            image,
            "-f",
            str(dockerfile),
            str(dockerfile.parent),
        )
        container = "monailabel-slicer-build-" + secrets.token_hex(8)
        run(
            "docker",
            "run",
            "-d",
            "--name",
            container,
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "--env",
            "HOME=/work/.home",
            "--volume",
            f"{root}:/work",
            image,
            "sleep",
            "infinity",
        )

        def execute(*command):
            run("docker", "exec", container, *command)

        try:
            configure = [
                "cmake",
                "-S",
                "/work/source-v5.12.4",
                "-B",
                BUILD,
                "-DCMAKE_BUILD_TYPE=Release",
                "-DSlicer_USE_SYSTEM_OpenSSL=ON",
                "-DBUILD_TESTING=OFF",
                "-DSlicer_BUILD_EXTENSIONMANAGER_SUPPORT=OFF",
                "-DSlicer_BUILD_APPLICATIONUPDATE_SUPPORT=OFF",
            ]
            # Bootstrap the native launcher before PythonSlicer can be generated.
            if not (root / "build-v5.12.4/CTKAppLauncherLib-build/bin/CTKAppLauncher").exists():
                execute(*configure)
                execute(
                    "cmake",
                    "--build",
                    BUILD,
                    "--target",
                    "CTKAppLauncherLib",
                    "--parallel",
                    str(args.jobs),
                )
            execute(*configure, f"-DCTKAppLauncher_DIR={BUILD}/CTKAppLauncherLib-build")
            execute("cmake", "--build", BUILD, "--parallel", str(args.jobs))
            execute(
                "cmake",
                "-S",
                "/work/dcmqi-v1.5.4",
                "-B",
                "/work/dcmqi-build",
                "-DCMAKE_BUILD_TYPE=Release",
                "-DDCMQI_SUPERBUILD=OFF",
                "-DBUILD_TESTING=OFF",
                "-DDCMQI_BUILD_DOC=OFF",
                f"-DITK_DIR={BUILD}/ITK-build",
                f"-DDCMTK_DIR={BUILD}/DCMTK-build",
                f"-DSlicerExecutionModel_DIR={BUILD}/SlicerExecutionModel-build",
            )
            execute("cmake", "--build", "/work/dcmqi-build", "--parallel", str(args.jobs))
            execute(
                f"{BUILD}/python-install/bin/PythonSlicer",
                "-m",
                "pip",
                "install",
                "--force-reinstall",
                "/work/dcmqi-0.4.0",
            )
            execute("cmake", "--build", f"{BUILD}/Slicer-build", "--target", "package")
        finally:
            with suppress(subprocess.SubprocessError):
                run("docker", "rm", "--force", container)
        packages = list((root / "build-v5.12.4/Slicer-build").glob("Slicer-*-linux-arm64.tar.gz"))
        if len(packages) != 1:
            raise RuntimeError("Expected one portable Slicer package in the build directory.")
        if destination.exists():
            raise RuntimeError(f"Build complete; move {destination} aside before publishing again.")
        with tempfile.TemporaryDirectory(dir=root, prefix=".publish-") as temporary:
            staging = Path(temporary) / "install"
            staging.mkdir()
            extract(packages[0], staging, "tar.gz")
            staging.rename(destination)
        show_installation(destination)


if __name__ == "__main__":
    main()
