#!/usr/bin/env python3
"""
comprehensive test script for the lightweight reviewer server.

Tests the new review-specific endpoints against the test data.
"""
import subprocess
import time
import requests
import json
import os
import sys
from pathlib import Path


TEST_DATA_DIR = os.environ.get("MONAI_LABEL_REVIEW_TEST_DATA", "/path/to/review-dataset")
PORT = 8079  # Different port from main monailabel server


def start_server():
    """Start the lightweight review server."""
    repo_root = Path(__file__).resolve().parents[1]
    print("=" * 70)
    print("Starting Lightweight Review Server")
    print("=" * 70)

    cmd = [
        sys.executable, "-m", "monailabel.main",
        "start_server",
        f"--app", "sample-apps/reviewer",
        f"--studies", f"{TEST_DATA_DIR}/images",
        f"--port", str(PORT),
        f"--conf", "mode", "review"
    ]

    print(f"Command: {' '.join(cmd)}")
    print()

    # Start server in background
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(repo_root)
    )

    # Wait for server to start
    print("Waiting for server to start...")

    # Check output for "READY TO ACCEPT REQUESTS" or timeout
    timeout = 30
    start_time = time.time()

    while time.time() - start_time < timeout:
        # Read from stdout
        try:
            output = proc.stdout.readline()
            if output:
                print(f"  {output.strip()}")

            # Check for prompt
            if "READY TO ACCEPT REQUESTS" in output:
                print("\n✅ Server started successfully!")
                break

        except:
            pass

        time.sleep(0.5)

    return proc


def wait_for_server_ready(proc):
    """Wait for server to be ready (check on another process)."""
    for _ in range(60):
        try:
            response = requests.get(f"http://localhost:{PORT}", timeout=2)
            if response.status_code == 200:
                print(f"✅ Server responding on port {PORT}")
                return True
        except:
            pass
        time.sleep(1)

    return False


def test_list_images():
    """Test listing images."""
    print("\n" + "=" * 70)
    print("TEST 1: List Images")
    print("=" * 70)

    try:
        response = requests.get(
            f"http://localhost:{PORT}/review/cases",
            params={"limit": 10},
            timeout=10
        )

        if response.status_code == 200:
            data = response.json()
            summary = data.get("summary", {})
            results = data.get("results", [])

            print(f"✅ Successfully listed images")
            print(f"   Total: {summary.get('total', 0)}")
            print(f"   Approved: {summary.get('approved', 0)}")
            print(f"   Flagged: {summary.get('flagged', 0)}")
            print(f"   Pending: {summary.get('pending', 0)}")
            print(f"\n   Sample images:")
            for img in results[:3]:
                print(f"      - {img.get('name', 'N/A')} (status: {img.get('status', 'N/A')})")

            # Check for review metadata
            if results and 'reviewer' in results[0]:
                print("   ✅ Review metadata present")
            else:
                print("   ⚠️  No review metadata in sample images")

            return True
        else:
            print(f"❌ List images failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False

    except Exception as e:
        print(f"❌ List images error: {e}")
        return False


def test_download_image():
    """Test downloading an image."""
    print("\n" + "=" * 70)
    print("TEST 2: Download Image")
    print("=" * 70)

    try:
        # Get a list of images first
        list_response = requests.get(
            f"http://localhost:{PORT}/review/cases",
            params={"limit": 1},
            timeout=10
        )

        if list_response.status_code != 200:
            print("⚠️  Skipped - cannot list images")
            return False

        images = list_response.json().get("results", [])
        if not images:
            print("⚠️  Skipped - no images available")
            return False

        image_id = images[0].get("id")

        # Download the image
        download_response = requests.get(
            f"http://localhost:{PORT}/datastore/image",
            params={"image": image_id},
            timeout=30
        )

        if download_response.status_code == 200:
            content_len = len(download_response.content)
            print(f"✅ Successfully downloaded image: {image_id}")
            print(f"   Size: {content_len:,} bytes ({content_len/1024/1024:.2f} MB)")
            return True
        else:
            print(f"❌ Download image failed: {download_response.status_code}")
            print(f"   Response: {download_response.text}")
            return False

    except Exception as e:
        print(f"❌ Download image error: {e}")
        return False


def test_download_label():
    """Test downloading a label."""
    print("\n" + "=" * 70)
    print("TEST 3: Download Label")
    print("=" * 70)

    try:
        # Get a list of images first
        list_response = requests.get(
            f"http://localhost:{PORT}/review/cases",
            params={"limit": 1},
            timeout=10
        )

        if list_response.status_code != 200:
            print("⚠️  Skipped - cannot list images")
            return False

        images = list_response.json().get("results", [])
        if not images:
            print("⚠️  Skipped - no images available")
            return False

        image_id = images[0].get("id")

        # Download the label with final tag
        download_response = requests.get(
            f"http://localhost:{PORT}/datastore/label",
            params={"label": image_id, "tag": "final"},
            timeout=10
        )

        if download_response.status_code == 200:
            print(f"✅ Successfully downloaded label: {image_id}")
            print(f"   Size: {len(download_response.content):,} bytes")
            print(f"   Content-Type: {download_response.headers.get('content-type', 'N/A')}")

            return True
        else:
            # Try without tag
            download_response = requests.get(
                f"http://localhost:{PORT}/datastore/label",
                params={"label": image_id},
                timeout=10
            )
            if download_response.status_code == 200:
                print(f"✅ Successfully downloaded label (default tag): {image_id}")
                print(f"   Size: {len(download_response.content):,} bytes")
                return True

            print(f"❌ Download label failed: {download_response.status_code}")
            print(f"   Response: {download_response.text}")
            return False

    except Exception as e:
        print(f"❌ Download label error: {e}")
        return False


def test_update_labelinfo():
    """Test updating label metadata."""
    print("\n" + "=" * 70)
    print("TEST 4: Update Label Info")
    print("=" * 70)

    try:
        list_response = requests.get(
            f"http://localhost:{PORT}/review/cases",
            params={"limit": 1},
            timeout=10
        )

        if list_response.status_code != 200:
            print("⚠️  Skipped - cannot list images")
            return False

        images = list_response.json().get("results", [])
        if not images:
            print("⚠️  Skipped - no images available")
            return False

        image_id = images[0].get("id")
        payload = {
            "info": json.dumps(
                {
                    "status": "approved",
                    "level": "medium",
                    "comment": "Manual smoke test",
                    "reviewer_name": "manual-smoke-test",
                }
            )
        }

        response = requests.put(
            f"http://localhost:{PORT}/datastore/label/info",
            params={"label": image_id, "tag": "final"},
            data=payload,
            timeout=10,
        )

        if response.status_code == 200:
            print(f"✅ Successfully updated label info for: {image_id}")
            return True

        print(f"❌ Update label info failed: {response.status_code}")
        print(f"   Response: {response.text}")
        return False

    except Exception as e:
        print(f"❌ Update label info error: {e}")
        return False


def test_generate_report():
    """Test generating review report."""
    print("\n" + "=" * 70)
    print("TEST 5: Generate Report")
    print("=" * 70)

    try:
        response = requests.get(
            f"http://localhost:{PORT}/review/report",
            params={"fmt": "json"},
            timeout=10,
        )

        if response.status_code == 200:
            data = response.json()
            print("✅ Successfully generated review report")
            print(f"   Keys: {sorted(data.keys())}")
            return True

        print(f"❌ Generate report failed: {response.status_code}")
        print(f"   Response: {response.text}")
        return False

    except Exception as e:
        print(f"❌ Generate report error: {e}")
        return False


def main():
    repo_root = Path(__file__).resolve().parents[1]
    os.chdir(repo_root)

    proc = start_server()
    try:
        if not wait_for_server_ready(proc):
            print("❌ Server did not become ready")
            return 1

        results = [
            test_list_images(),
            test_download_image(),
            test_download_label(),
            test_update_labelinfo(),
            test_generate_report(),
        ]
        return 0 if all(results) else 1
    finally:
        proc.terminate()
        proc.wait(timeout=10)


if __name__ == "__main__":
    raise SystemExit(main())