"""
Script to create the first admin user.
Run this script to create an initial admin account that can then register other users.
"""

from auth import hash_password
from models import User
from database import init_database

def create_first_admin():
    # Initialize database tables if they don't exist
    init_database()
    
    print("=" * 50)
    print("Create First Admin User")
    print("=" * 50)
    
    # Get admin details
    name = input("Enter admin name: ")
    email = input("Enter admin email: ")
    password = input("Enter admin password (min 6 characters): ")
    image_url = input("Enter image URL (optional, press Enter to skip): ").strip() or None
    
    # Validate password length
    if len(password) < 6:
        print("\n❌ Error: Password must be at least 6 characters long")
        return
    
    # Check if email already exists
    if User.email_exists(email):
        print(f"\n❌ Error: Email '{email}' is already registered")
        return
    
    # Hash password and create user
    hashed_password = hash_password(password)
    
    try:
        user_id = User.create(
            name=name,
            email=email,
            password=hashed_password,
            image_url=image_url,
            role="admin"
        )
        
        print("\n✅ Admin user created successfully!")
        print(f"User ID: {user_id}")
        print(f"Name: {name}")
        print(f"Email: {email}")
        print(f"Role: admin")
        print("\nYou can now use this account to login and register other users.")
        
    except Exception as e:
        print(f"\n❌ Error creating admin user: {e}")

if __name__ == "__main__":
    create_first_admin()
