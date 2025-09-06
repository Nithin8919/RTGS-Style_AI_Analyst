#!/usr/bin/env python3
"""
Test script to verify Groq configuration is working properly
"""
import os
import yaml
from groq import Groq
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def test_groq_config():
    """Test Groq configuration and API connectivity"""
    print("Testing Groq configuration...")
    
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Check if GROQ_API_KEY is set
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        print("❌ GROQ_API_KEY environment variable is not set!")
        print("Please set your Groq API key:")
        print("export GROQ_API_KEY='your-api-key-here'")
        return False
    
    print(f"✅ GROQ_API_KEY is set: {api_key[:10]}...")
    
    # Check configuration
    groq_config = config.get('groq', {})
    if not groq_config:
        print("❌ Groq configuration not found in config.yaml")
        return False
    
    print(f"✅ Groq model: {groq_config.get('model', 'Not set')}")
    print(f"✅ Max tokens: {groq_config.get('max_tokens', 'Not set')}")
    print(f"✅ Temperature: {groq_config.get('temperature', 'Not set')}")
    
    # Test API connection
    try:
        client = Groq(api_key=api_key)
        
        # Simple test call
        response = client.chat.completions.create(
            model=groq_config.get('model', 'llama-3.1-8b-instant'),
            messages=[
                {"role": "user", "content": "Hello! Please respond with 'Groq is working!' to confirm the connection."}
            ],
            max_tokens=50,
            temperature=0.1
        )
        
        print("✅ Groq API connection successful!")
        print(f"Response: {response.choices[0].message.content}")
        return True
        
    except Exception as e:
        print(f"❌ Groq API connection failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_groq_config()
    if success:
        print("\n🎉 Groq configuration is working perfectly!")
    else:
        print("\n⚠️  Please fix the issues above before running the RTGS AI Analyst")
