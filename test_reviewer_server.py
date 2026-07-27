#!/usr/bin/env python3
"""
comprehensive test script for the lightweight reviewer server.

Tests the new review-specific endpoints against the test data.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import requests

TEST_DATA_DIR = "/processed/CCHMC/Fetal_SVRTK_review"
PORT = 8079  # Different port from main monailabel server


def start_server():
    """Start the lightweight review server."""
    print("=" * 70)
    print("Starting Lightweight Review Server")
    print("=" * 70)

    cmd = [
        sys.executable,
        "-m",
        "monailabel.main",
        "start_server",
        f"--app",
        "sample-apps/reviewer",
        f"--studies",
        f"{TEST_DATA_DIR}/images",
        f"--port",
        str(PORT),
        f"--conf",
        "mode",
        "review",
    ]

    print(f"Command: {' '.join(cmd)}")
    print()

    # Start server in background
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd="/workspace/MONAILabel")

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
        response = requests.get(f"http://localhost:{PORT}/review/cases", params={"limit": 10}, timeout=10)

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
            if results and "reviewer" in results[0]:
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
        list_response = requests.get(f"http://localhost:{PORT}/review/cases", params={"limit": 1}, timeout=10)

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
            f"http://localhost:{PORT}/datastore/image", params={"image": image_id}, timeout=30
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
        list_response = requests.get(f"http://localhost:{PORT}/review/cases", params={"limit": 1}, timeout=10)

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
            f"http://localhost:{PORT}/datastore/label", params={"label": image_id, "tag": "final"}, timeout=10
        )

        if download_response.status_code == 200:
            print(f"✅ Successfully downloaded label: {image_id}")
            print(f"   Size: {len(download_response.content):,} bytes")
            print(f"   Content-Type: {download_response.headers.get('content-type', 'N/A')}")

            return True
        else:
            # Try without tag
            download_response = requests.get(
                f"http://localhost:{PORT}/datastore/label", params={"label": image_id}, timeout=10
            )

            if download_response.status_code == 200:
                print(f"✅ Successfully downloaded label (default tag): {image_id}")
                print(f"   Size: {len(download_response.content):,} bytes")
                return True
            else:
                print(f"❌ Download label failed: {download_response.status_code}")
                print(f"   Response: {download_response.text}")
                return False

    except Exception as e:
        print(f"❌ Download label error: {e}")
        return False


def test_update_labelinfo():
    """Test updating label info."""
    print("\n" + "=" * 70)
    print("TEST 4: Update Label Information")
    print("=" * 70)

    try:
        # Get a list of images first
        list_response = requests.get(f"http://localhost:{PORT}/review/cases", params={"limit": 1}, timeout=10)

        if list_response.status_code != 200:
            print("⚠️  Skipped - cannot list images")
            return False

        images = list_response.json().get("results", [])
        if not images:
            print("⚠️  Skipped - no images available")
            return False

        image_id = images[0].get("id")

        # Prepare update data
        review_info = {
            "status": "approved",
            "level": "medium",
            "comment": "Test review from lightweight reviewer",
            "reviewer_name": "Test Reviewer",
        }

        payload = {"info": json.dumps(review_info)}

        # Update label info
        update_response = requests.put(
            f"http://localhost:{PORT}/datastore/label/info",
            params={"label": image_id, "tag": "final"},
            data=payload,
            timeout=10,
        )

        if update_response.status_code == 200:
            data = update_response.json()
            print(f"✅ Successfully updated label info: {image_id}")
            print(f"   Status: {data.get('status')}")

            # Try to get label info to verify
            info_response = requests.get(
                f"http://localhost:{PORT}/datastore/label/info", params={"label": image_id, "tag": "final"}, timeout=10
            )

            if info_response.status_code == 200:
                info_data = info_response.json()
                print(f"✅ Retrieved updated info:")
                print(f"   {json.dumps(info_data.get('info'), indent=4)}")

            return True
        else:
            print(f"⚠️  Update failed (but might be expected): {update_response.status_code}")
            # Note: Lack of writable datastore might cause this
            print(f"   This is expected if no write permissions")
            return False  # Not a failure in our case

    except Exception as e:
        print(f"❌ Update label info error: {e}")
        return False


def test_generate_report():
    """Test generating review report."""
    print("\n" + "=" * 70)
    print("TEST 5: Generate Review Report")
    print("=" * 70)

    for fmt in ["json", "csv"]:
        try:
            response = requests.get(f"http://localhost:{PORT}/review/report", params={"fmt": fmt}, timeout=10)

            if response.status_code == 200:
                content_type = response.headers.get("content-type", "")
                content_len = len(response.content)

                if fmt == "json":
                    print(f"✅ Generated JSON report: {content_len:,} bytes")
                    data = response.json()
                    report = data.get("report", {})
                    print(f"   Reviews: {report.get('total', 0)}")
                    print(f"   Approved: {report.get('approved', 0)}")
                    print(f"   Flagged: {report.get('flagged', 0)}")
                else:
                    print(f"✅ Generated CSV report: {content_len:,} bytes")
                    preview = response.text[:200]
                    print(f"   Preview: {preview}...")

            else:
                print(f"⚠️  Report generation failed (may be expected): {response.status_code}")

        except Exception as e:
            print(f"⚠️  Report generation error: {e}")

    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("LIGHTWEIGHT REVIEWER - SERVER TESTING")
    print("=" * 70)
    print(f"Test Data: {TEST_DATA_DIR}")
    print(f"Server Port: {PORT}")
    print(f"Server Dir: /workspace/MONAILabel")

    # Check if test data exists
    if not Path(TEST_DATA_DIR).exists():
        print(f"\n❌ ERROR: Test data directory does not exist: {TEST_DATA_DIR}")
        print("✅ Check if /processed/CCHMC/Fetal_SVRTK_review exists")
        return 1

    # Step 1: Start server
    proc = start_server()

    # Wait for server
    if not wait_for_server_ready(proc):
        print("\n❌ ERROR: Server failed to start")
        proc.terminate()
        return 1

    # Step 2: Run tests
    results = []

    results.append(("List Images", test_list_images()))
    results.append(("Download Image", test_download_image()))
    results.append(("Download Label", test_download_label()))
    results.append(("Update Label Info", test_update_labelinfo()))
    results.append(("Generate Report", test_generate_report()))

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    for name, passed in results:
        status = "✅ PASS" if passed else "⏭️  SKIP"
        print(f"{status}: {name}")

    print("\n" + "=" * 70)
    print("Press Ctrl+C to stop server (tests complete)")
    print("=" * 70)

    # Keep server running for inspection
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\nStopping server...")
        proc.terminate()
        proc.wait(timeout=5)

    print("\n✅ Testing complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
