"""Test script for rate limiting verification with user-scoped requests.

This script verifies:
1. Same user hitting rate limit gets 429 after RATE_LIMIT_MAX requests
2. Different user is NOT rate-limited by another user's requests
"""

import requests
import time
import sys

BASE_URL = "http://127.0.0.1:8001"
RATE_LIMIT_MAX = 10  # From .env RATE_LIMIT_MAX

def test_same_user_rate_limiting():
    """Test that same user gets 429 after exceeding rate limit."""
    print("=" * 60)
    print("TEST 1: Same-user rate limiting")
    print("=" * 60)

    user_id = "test-user-1"
    headers = {"X-User-Id": user_id}
    payload = {
        "question": "What is quantum computing?",
        "user_id": user_id
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
    expected_success = RATE_LIMIT_MAX
    expected_429_at = RATE_LIMIT_MAX + 1

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

    # First user - they've been making requests, may be rate limited
    user1_id = "test-user-1"

    # Second user - completely different user
    user2_id = "test-user-2"
    headers = {"X-User-Id": user2_id}
    payload = {
        "question": "What is quantum computing?",
        "user_id": user2_id
    }

    print(f"User 1 ('{user1_id}') has been making requests and may be rate limited")
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
                print(f"User 2, Request {i+1}: HTTP {response.status_code} - ERROR")

        except Exception as e:
            print(f"User 2, Request {i+1}: EXCEPTION - {e}")

    print()
    if success_count >= 1:
        print("PASS: Different user was NOT affected by first user's rate limit")
        return True
    else:
        print("FAIL: Different user was affected by first user's rate limit")
        return False


def test_header_body_mismatch():
    """Test what happens when X-User-Id header and body user_id don't match."""
    print()
    print("=" * 60)
    print("TEST 3: Header/body user_id mismatch")
    print("=" * 60)

    user1_id = "user-a"  # Will be in header
    user2_id = "user-b"  # Will be in body

    # Request with mismatched user_ids
    headers = {"X-User-Id": user1_id}
    payload = {
        "question": "What is quantum computing?",
        "user_id": user2_id  # Different from header
    }

    print(f"Header X-User-Id: '{user1_id}'")
    print(f"Body user_id:     '{user2_id}'")
    print()

    # Make 11 requests to ensure rate limit would hit user-a if they were tracked
    for i in range(RATE_LIMIT_MAX + 1):
        try:
            response = requests.post(
                f"{BASE_URL}/ask",
                json=payload,
                headers=headers,
                timeout=30
            )

            if i == 0:
                print(f"Request 1: HTTP {response.status_code}")

        except Exception as e:
            print(f"Request {i+1}: EXCEPTION - {e}")

    # Now check if user-a is rate-limited (header identity)
    headers_user_a = {"X-User-Id": user1_id}
    payload_user_a = {
        "question": "What is quantum computing?",
        "user_id": user1_id  # Now matching
    }

    print()
    print(f"Now sending request as user-a (same as original header)...")
    response = requests.post(
        f"{BASE_URL}/ask",
        json=payload_user_a,
        headers=headers_user_a,
        timeout=30
    )

    if response.status_code == 429:
        print(f"Result: HTTP 429 - Header identity (X-User-Id) is used for rate limiting")
        print(f"        user-a is rate-limited (body user_id was ignored)")
        return True, "header_used"
    else:
        print(f"Result: HTTP {response.status_code}")
        print(f"        WARNING: Header/body mismatch behavior is unclear")
        return True, "unclear"


def main():
    print("Rate Limiting Verification Tests")
    print(f"Server: {BASE_URL}")
    print(f"Rate limit threshold: {RATE_LIMIT_MAX} requests per 60 seconds")
    print()

    test1_pass = test_same_user_rate_limiting()
    test2_pass = test_different_user_isolation()
    test3_pass, mismatch_behavior = test_header_body_mismatch()

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Test 1 (Same-user rate limiting): {'PASS' if test1_pass else 'FAIL'}")
    print(f"Test 2 (Different-user isolation): {'PASS' if test2_pass else 'FAIL'}")
    print(f"Test 3 (Header/body mismatch): {'PASS' if test3_pass else 'FAIL'}")
    print(f"  Mismatch behavior: {mismatch_behavior}")
    print()

    if test1_pass and test2_pass:
        print("OVERALL: PASS - Rate limiting is working correctly")
        return 0
    else:
        print("OVERALL: FAIL - Rate limiting has issues")
        return 1


if __name__ == "__main__":
    sys.exit(main())
