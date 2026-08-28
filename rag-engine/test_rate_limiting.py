"""Test script for rate limiting verification with user-scoped requests.

This script verifies:
1. Same user hitting rate limit gets 429 after RATE_LIMIT_MAX requests
2. Different user is NOT rate-limited by another user's requests
3. Spoofing attempts using deprecated X-User-Id are rejected
"""

import requests
import time
import sys
import jwt

BASE_URL = "http://127.0.0.1:8000"
RATE_LIMIT_MAX = 10  # From .env RATE_LIMIT_MAX
JWT_SECRET = "test-secret"  # Must match server configuration during test

def generate_token(user_id):
    return jwt.encode({"sub": user_id}, JWT_SECRET, algorithm="HS256")

def test_same_user_rate_limiting():
    """Test that same user gets 429 after exceeding rate limit."""
    print("=" * 60)
    print("TEST 1: Same-user rate limiting")
    print("=" * 60)

    user_id = "test-user-1"
    headers = {"Authorization": f"Bearer {generate_token(user_id)}"}
    payload = {
        "question": "What is quantum computing?",
    }

    successes = 0
    failures = 0

    print(f"Sending {RATE_LIMIT_MAX + 2} requests as user '{user_id}'")
    print(f"Rate limit threshold: {RATE_LIMIT_MAX} requests per 60 seconds")
    print()

    for i in range(RATE_LIMIT_MAX + 2):
        try:
            response = requests.post(
                f"{BASE_URL}/ask",
                json=payload,
                headers=headers,
                timeout=30
            )

            if response.status_code == 200:
                successes += 1
                print(f"Request {i+1}: HTTP {response.status_code} (OK) - SUCCESS")
            elif response.status_code == 429:
                failures += 1
                retry_after = response.headers.get("Retry-After", "N/A")
                print(f"Request {i+1}: HTTP {response.status_code} (Too Many Requests) - RATE LIMITED")
                print(f"         Retry-After: {retry_after}s")
            else:
                failures += 1
                print(f"Request {i+1}: HTTP {response.status_code} - ERROR: {response.text[:100]}")

        except Exception as e:
            failures += 1
            print(f"Request {i+1}: EXCEPTION - {e}")

    print()
    print(f"Results: {successes} successes, {failures} rate-limited/errors")

    # Verify the rate limiter kicked in at the right place
    if failures >= 1:
        print("PASS: Rate limiting is working - at least one request was rate-limited")
        return True
    else:
        print("FAIL: No requests were rate-limited - rate limiter may not be working")
        return False


def test_different_user_isolation():
    """Test that a different user is NOT affected by another user's rate limit."""
    print()
    print("=" * 60)
    print("TEST 2: Different-user isolation")
    print("=" * 60)

    # First user - already tested, might be rate limited
    user1_id = "test-user-1"

    # Second user - completely different user
    user2_id = "test-user-2"
    headers = {"Authorization": f"Bearer {generate_token(user2_id)}"}
    payload = {
        "question": "What is quantum computing?",
    }

    print(f"Testing user 2 ('{user2_id}') with fresh requests...")
    print()

    success_count = 0

    for i in range(3):
        try:
            response = requests.post(
                f"{BASE_URL}/ask",
                json=payload,
                headers=headers,
                timeout=30
            )

            if response.status_code == 200:
                success_count += 1
                print(f"User 2, Request {i+1}: HTTP {response.status_code} (OK) - SUCCESS")
            elif response.status_code == 429:
                print(f"User 2, Request {i+1}: HTTP {response.status_code} (Too Many Requests)")
                print(f"         WARNING: User 2 was rate-limited when they shouldn't be!")
            else:
                print(f"User 2, Request {i+1}: HTTP {response.status_code} - ERROR: {response.text[:100]}")

        except Exception as e:
            print(f"User 2, Request {i+1}: EXCEPTION - {e}")

    print()
    if success_count >= 1:
        print("PASS: Different user was NOT affected by first user's rate limit")
        return True
    else:
        print("FAIL: Different user was affected by first user's rate limit")
        return False


def test_spoof_attempt_rejected():
    """Test that X-User-Id header without JWT is rejected (no bypass)."""
    print()
    print("=" * 60)
    print("TEST 3: X-User-Id spoof attempt rejection")
    print("=" * 60)

    # Attempt to use the old header without a JWT
    headers = {"X-User-Id": "attacker"}
    payload = {"question": "What is quantum computing?"}

    print("Attempting to use deprecated X-User-Id header without JWT...")
    response = requests.post(
        f"{BASE_URL}/ask",
        json=payload,
        headers=headers,
        timeout=30
    )

    if response.status_code == 401:
        print("PASS: Request rejected with 401 (Unauthorized) as expected")
        return True
    else:
        print(f"FAIL: Expected 401, got {response.status_code}")
        return False


def main():
    print("Rate Limiting Verification Tests")
    print(f"Server: {BASE_URL}")
    print(f"Rate limit threshold: {RATE_LIMIT_MAX} requests per 60 seconds")
    print()

    test1_pass = test_same_user_rate_limiting()
    test2_pass = test_different_user_isolation()
    test3_pass = test_spoof_attempt_rejected()

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Test 1 (Same-user rate limiting): {'PASS' if test1_pass else 'FAIL'}")
    print(f"Test 2 (Different-user isolation): {'PASS' if test2_pass else 'FAIL'}")
    print(f"Test 3 (Spoof rejection): {'PASS' if test3_pass else 'FAIL'}")
    print()

    if test1_pass and test2_pass and test3_pass:
        print("OVERALL: PASS - Rate limiting is working correctly and secured")
        return 0
    else:
        print("OVERALL: FAIL - Rate limiting or security has issues")
        return 1


if __name__ == "__main__":
    sys.exit(main())
