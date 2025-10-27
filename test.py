"""
Simple script to test NyayaBot API endpoints.
Run this after starting the server to verify everything works.

Usage:
    python test_api.py
"""

import requests
import json
from typing import Dict

API_BASE_URL = "http://localhost:8000"


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def test_root():
    """Test root endpoint."""
    print_section("Testing Root Endpoint")
    response = requests.get(f"{API_BASE_URL}/")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200


def test_health():
    """Test health check endpoint."""
    print_section("Testing Health Check")
    response = requests.get(f"{API_BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200


def test_languages():
    """Test supported languages endpoint."""
    print_section("Testing Supported Languages")
    response = requests.get(f"{API_BASE_URL}/languages")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200


def test_question(query: str, language: str = None, description: str = ""):
    """Test asking a question."""
    print_section(f"Testing Question: {description}")
    print(f"Query: {query}")
    print(f"Language: {language or 'auto-detect'}")
    
    payload = {
        "query": query,
        "top_k": 3,
        "include_sources": True
    }
    
    if language:
        payload["language"] = language
    
    response = requests.post(f"{API_BASE_URL}/ask", json=payload)
    print(f"\nStatus: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\nDetected Language: {result.get('language')}")
        print(f"\nAnswer:\n{result.get('answer')}")
        
        if result.get('sources'):
            print(f"\nTop Sources:")
            for i, source in enumerate(result['sources'][:2], 1):
                print(f"\n  {i}. {source.get('source')} (Relevance: {source.get('relevance_score')})")
                print(f"     {source.get('text')[:150]}...")
        
        return True
    else:
        print(f"Error: {response.text}")
        return False


def test_batch_questions():
    """Test batch question endpoint."""
    print_section("Testing Batch Questions")
    
    payload = {
        "queries": [
            "What is Article 14?",
            "What is Article 21?",
            "What is Right to Freedom?"
        ],
        "language": "en"
    }
    
    response = requests.post(f"{API_BASE_URL}/batch-ask", json=payload)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        results = response.json()["results"]
        print(f"\nProcessed {len(results)} questions")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result.get('original_query')}")
            print(f"   Answer: {result.get('answer')[:100]}...")
        return True
    else:
        print(f"Error: {response.text}")
        return False


def main():
    """Run all tests."""
    print("\n" + "🔧 " * 20)
    print("  NYAYABOT API TEST SUITE")
    print("🔧 " * 20)
    
    tests = []
    
    # Basic endpoint tests
    tests.append(("Root Endpoint", test_root()))
    tests.append(("Health Check", test_health()))
    tests.append(("Supported Languages", test_languages()))
    
    # Question tests in different languages
    tests.append((
        "English Question",
        test_question(
            "What are fundamental rights in Indian Constitution?",
            "en",
            "English - Fundamental Rights"
        )
    ))
    
    tests.append((
        "Hindi Question",
        test_question(
            "भारतीय संविधान में मौलिक अधिकार क्या हैं?",
            "hi",
            "Hindi - Fundamental Rights"
        )
    ))
    
    tests.append((
        "Marathi Question",
        test_question(
            "भारतीय संविधानातील मूलभूत अधिकार कोणते आहेत?",
            "mr",
            "Marathi - Fundamental Rights"
        )
    ))
    
    # Batch questions
    tests.append(("Batch Question Test", test_batch_questions()))
    
    # Summary
    print_section("Test Summary")
    for name, result in tests:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{name:<30} : {status}")

    print("\nAll tests completed.")


if __name__ == "__main__":
    main()

