"""
Example usage of the Authentication API
This script demonstrates how to interact with the API endpoints
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_login(email: str, password: str):
    """Test the login endpoint"""
    print("\n" + "="*50)
    print("Testing Login")
    print("="*50)
    
    response = requests.post(
        f"{BASE_URL}/auth/login",
        json={
            "email": email,
            "password": password
        }
    )
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print("✅ Login successful!")
        print(f"Token: {data['access_token'][:50]}...")
        print(f"User: {data['user']['name']} ({data['user']['role']})")
        return data['access_token']
    else:
        print("❌ Login failed!")
        print(f"Error: {response.json()}")
        return None

def test_signup(token: str, user_data: dict):
    """Test the signup endpoint (admin only)"""
    print("\n" + "="*50)
    print("Testing Signup")
    print("="*50)
    
    response = requests.post(
        f"{BASE_URL}/auth/signup",
        json=user_data,
        headers={
            "Authorization": f"Bearer {token}"
        }
    )
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 201:
        data = response.json()
        print("✅ Signup successful!")
        print(f"New User: {data['name']} ({data['role']})")
        print(f"Email: {data['email']}")
        print(f"ID: {data['id']}")
    else:
        print("❌ Signup failed!")
        print(f"Error: {response.json()}")

def main():
    print("Authentication API Test Script")
    print("Make sure the API is running on http://localhost:8000")
    
    # Example 1: Admin Login
    admin_token = test_login(
        email="admin@example.com",
        password="admin123"
    )
    
    if admin_token:
        # Example 2: Register a new chief
        test_signup(
            token=admin_token,
            user_data={
                "name": "John Chef",
                "email": "chef1@example.com",
                "password": "chef123456",
                "image_url": "https://example.com/chef1.jpg",
                "role": "chief"
            }
        )
        
        # Example 3: Register a new admin
        test_signup(
            token=admin_token,
            user_data={
                "name": "Jane Admin",
                "email": "admin2@example.com",
                "password": "admin123456",
                "image_url": "https://example.com/admin2.jpg",
                "role": "admin"
            }
        )
    
    # Example 4: Login as newly created chief
    test_login(
        email="chef1@example.com",
        password="chef123456"
    )
    
    print("\n" + "="*50)
    print("Test completed!")
    print("="*50)

if __name__ == "__main__":
    main()
