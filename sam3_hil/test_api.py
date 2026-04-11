#!/usr/bin/env python3
"""
STAMP API Test Script
=====================

Simple script to test if the API is working.

Usage:
    python test_api.py                          # Test localhost:8000
    python test_api.py --url http://192.168.1.100:8000
    python test_api.py --image /path/to/image.jpg --prompt "ship"
"""

import argparse
import base64
import sys
from pathlib import Path

try:
    import requests
except ImportError:
    print("Please install requests: pip install requests")
    sys.exit(1)


def test_health(base_url: str) -> bool:
    """Test health endpoint."""
    print(f"\n[1] Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print(f"    ✅ Health check passed: {response.json()}")
            return True
        else:
            print(f"    ❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"    ❌ Connection failed: {e}")
        return False


def test_root(base_url: str) -> bool:
    """Test root endpoint."""
    print(f"\n[2] Testing root endpoint...")
    try:
        response = requests.get(f"{base_url}/", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"    ✅ Service: {data.get('service')}")
            print(f"    ✅ Status: {data.get('status')}")
            print(f"    ✅ SAM3 loaded: {data.get('sam3_loaded')}")
            return True
        else:
            print(f"    ❌ Root check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"    ❌ Connection failed: {e}")
        return False


def test_detect(base_url: str, image_path: str, prompt: str) -> bool:
    """Test detection endpoint."""
    print(f"\n[3] Testing detection endpoint...")
    
    if not Path(image_path).exists():
        print(f"    ⚠️  Image not found: {image_path}")
        print(f"    ⚠️  Skipping detection test")
        return True
    
    try:
        with open(image_path, "rb") as f:
            files = {"image": f}
            data = {"prompt": prompt}
            response = requests.post(
                f"{base_url}/api/detect",
                files=files,
                data=data,
                timeout=60,
            )
        
        if response.status_code == 200:
            result = response.json()
            print(f"    ✅ Detection successful")
            print(f"    ✅ Found {result.get('num_detections', 0)} objects")
            for det in result.get('detections', [])[:3]:
                print(f"       - Object {det['obj_id']}: score={det['score']:.2f}, category={det['category']}")
            return True
        else:
            print(f"    ❌ Detection failed: {response.status_code}")
            print(f"       {response.text}")
            return False
            
    except Exception as e:
        print(f"    ❌ Detection error: {e}")
        return False


def test_docs(base_url: str) -> bool:
    """Test API docs endpoint."""
    print(f"\n[4] Testing API docs...")
    try:
        response = requests.get(f"{base_url}/docs", timeout=5)
        if response.status_code == 200:
            print(f"    ✅ API docs available at: {base_url}/docs")
            return True
        else:
            print(f"    ❌ Docs not available: {response.status_code}")
            return False
    except Exception as e:
        print(f"    ❌ Docs error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test STAMP API")
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000",
        help="API base URL",
    )
    parser.add_argument(
        "--image",
        type=str,
        default="test.jpg",
        help="Test image path",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="object",
        help="Detection prompt",
    )
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("STAMP API Test")
    print("=" * 50)
    print(f"Target: {args.url}")
    
    results = []
    results.append(("Health", test_health(args.url)))
    results.append(("Root", test_root(args.url)))
    results.append(("Docs", test_docs(args.url)))
    results.append(("Detect", test_detect(args.url, args.image, args.prompt)))
    
    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    
    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
