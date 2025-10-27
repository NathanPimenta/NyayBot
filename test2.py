"""
Extended NyayaBot API Test Suite
Includes complex multilingual and contextual legal queries.
"""

import requests
import json

API_BASE_URL = "http://localhost:8000"

def print_section(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def test_question(query, language=None, description=""):
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
            for i, source in enumerate(result['sources'][:3], 1):
                print(f"\n  {i}. {source.get('source')} (Relevance: {source.get('relevance_score')})")
                print(f"     {source.get('text')[:200]}...")
    else:
        print(f"Error: {response.text}")


def main():
    print_section("🔧 NYAYABOT COMPLEX QUERY TEST SUITE 🔧")

    # ------------------ SIMPLE QUERIES ------------------
    test_question(
        "What are fundamental rights in Indian Constitution?",
        "en",
        "English - Simple Query"
    )

    # ------------------ COMPLEX QUERIES ------------------
    test_question(
        "Can the government restrict my freedom of speech during national emergencies?",
        "en",
        "English - Conditional Rights Query"
    )

    test_question(
        "What protections do I have if I am detained without being told the reason?",
        "en",
        "English - Arrest and Detention Rights"
    )

    test_question(
        "भारतीय संविधान के अनुसार समानता का अधिकार किन परिस्थितियों में सीमित किया जा सकता है?",
        "hi",
        "Hindi - Restrictions on Right to Equality"
    )

    test_question(
        "जर पोलिस वॉरंट शिवाय घरात येऊन अटक करत असतील, तर नागरिकाचे कोणते अधिकार आहेत?",
        "mr",
        "Marathi - Police Arrest without Warrant"
    )

    test_question(
        "What is the difference between Directive Principles and Fundamental Rights?",
        "en",
        "English - Conceptual Comparison"
    )

    test_question(
        "Do I have the right to privacy under the Indian Constitution?",
        "en",
        "English - Modern Right (Article 21)"
    )

    test_question(
        "माझ्यावर चुकीच्या आरोपाखाली खटला दाखल झाला असल्यास मला न्याय मिळवण्यासाठी काय करता येईल?",
        "mr",
        "Marathi - Fair Trial Rights"
    )

    test_question(
        "क्या राज्य सरकार नागरिकों के धार्मिक स्वतंत्रता के अधिकार को सीमित कर सकती है?",
        "hi",
        "Hindi - Freedom of Religion Restrictions"
    )


if __name__ == "__main__":
    main()

