#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to encode Firebase credentials from JSON file to base64 string
and save it to .env file.
"""

import base64
import os
import json
from dotenv import load_dotenv

def encode_firebase_credentials():
    """Encode Firebase credentials to base64 and save to .env file"""
    # Path to credentials file
    creds_path = os.path.join(os.path.dirname(__file__), "firebase-credentials.json")
    
    if not os.path.exists(creds_path):
        print(f"Error: Firebase credentials file not found at: {creds_path}")
        return False
        
    try:
        # Read and encode credentials
        with open(creds_path, 'r') as f:
            creds_json = f.read()
        
        # Encode to base64
        creds_base64 = base64.b64encode(creds_json.encode('utf-8')).decode('utf-8')
        
        # Check if .env exists and read existing content
        env_path = os.path.join(os.path.dirname(__file__), ".env")
        existing_env = {}
        if os.path.exists(env_path):
            load_dotenv(env_path)
            with open(env_path, 'r') as f:
                for line in f:
                    if '=' in line:
                        key, value = line.strip().split('=', 1)
                        existing_env[key] = value
        
        # Add or update credentials in env
        existing_env['FIREBASE_CREDENTIALS_BASE64'] = creds_base64
        
        # Write back to .env file
        with open(env_path, 'w') as f:
            for key, value in existing_env.items():
                f.write(f"{key}={value}\n")
        
        print("✅ Success: Firebase credentials have been encoded and saved to .env file")
        print("You can now delete the firebase-credentials.json file for security")
        return True
        
    except Exception as e:
        print(f"Error encoding credentials: {str(e)}")
        return False

if __name__ == "__main__":
    encode_firebase_credentials()
