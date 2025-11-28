# FastAPI Authentication System with MySQL

A complete authentication system built with FastAPI and MySQL, featuring user registration, login, and login history tracking.

## Features

- ✅ User registration (admin and chief roles)
- ✅ Admin-only signup (only admins can register new users)
- ✅ Secure password hashing with bcrypt
- ✅ JWT token-based authentication
- ✅ Login history tracking (IP address, user agent, timestamp)
- ✅ Role-based access control
- ✅ User profile with image URL support
- ✅ CORS enabled for frontend integration

## User Attributes

- `id`: Auto-incrementing primary key
- `name`: User's full name
- `email`: Unique email address
- `password`: Hashed password
- `image_url`: URL to user's profile photo
- `role`: Either 'admin' or 'chief'
- `created_at`: Account creation timestamp

## Installation

1. **Install Python dependencies:**
```bash
pip install -r requirements.txt
```

2. **Setup MySQL database:**
```sql
CREATE DATABASE ai_db;
```

3. **Initialize database tables:**
```bash
python database.py
```

## Configuration

The system uses the following database credentials (configured in `database.py`):
- Host: `localhost`
- User: `root`
- Password: `1234`
- Database: `ai_db`

To change these, edit the `DB_CONFIG` in `database.py`.

## Running the Application

```bash
python main.py
```

Or with uvicorn directly:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at: `http://localhost:8000`

## API Documentation

Once running, visit:
- Interactive API docs: `http://localhost:8000/docs`
- Alternative docs: `http://localhost:8000/redoc`

## API Endpoints

### 1. **Signup (Admin Only)**
**POST** `/auth/signup`

Register a new user (admin or chief). Requires admin authentication.

**Headers:**
```
Authorization: Bearer <admin_access_token>
```

**Request Body:**
```json
{
  "name": "John Doe",
  "email": "john@example.com",
  "password": "securepassword123",
  "image_url": "https://example.com/photo.jpg",
  "role": "chief"
}
```

**Response:**
```json
{
  "id": 1,
  "name": "John Doe",
  "email": "john@example.com",
  "image_url": "https://example.com/photo.jpg",
  "role": "chief",
  "created_at": "2025-11-28T10:30:00"
}
```

### 2. **Login**
**POST** `/auth/login`

Authenticate a user and receive an access token. Also creates a login history record.

**Request Body:**
```json
{
  "email": "john@example.com",
  "password": "securepassword123"
}
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "user": {
    "id": 1,
    "name": "John Doe",
    "email": "john@example.com",
    "image_url": "https://example.com/photo.jpg",
    "role": "chief",
    "created_at": "2025-11-28T10:30:00"
  }
}
```

## Database Schema

### Users Table
```sql
CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password VARCHAR(255) NOT NULL,
    image_url TEXT,
    role ENUM('admin', 'chief') NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Login History Table
```sql
CREATE TABLE login_history (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    login_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ip_address VARCHAR(45),
    user_agent TEXT,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);
```

## Authentication Flow

1. **Initial Setup**: Create the first admin user directly in the database:
```sql
INSERT INTO users (name, email, password, role) 
VALUES ('Admin', 'admin@example.com', '<hashed_password>', 'admin');
```

2. **Admin Login**: Admin logs in and receives a JWT token
3. **Register Users**: Admin uses the token to register new users (admin or chief)
4. **User Login**: Registered users can log in and receive their own tokens
5. **Login Tracking**: Every login is recorded in the `login_history` table

## Security Features

- **Password Hashing**: Passwords are hashed using bcrypt before storage
- **JWT Tokens**: Secure token-based authentication with 24-hour expiration
- **Role-Based Access**: Signup endpoint restricted to admin users only
- **Login Tracking**: All login attempts are recorded with IP and user agent

## File Structure

```
.
├── main.py           # FastAPI application entry point
├── database.py       # Database connection and initialization
├── models.py         # Database models (User, LoginHistory)
├── schemas.py        # Pydantic schemas for request/response validation
├── auth.py           # Authentication utilities (JWT, password hashing)
├── routes.py         # API routes (signup, login)
├── requirements.txt  # Python dependencies
└── README.md         # This file
```

## Frontend Integration

When integrating with your frontend:

1. **Signup**: Send admin token in Authorization header
2. **Login**: Send email and password, receive token
3. **Store Token**: Save the JWT token in localStorage or secure storage
4. **Authenticated Requests**: Include token in Authorization header:
   ```
   Authorization: Bearer <access_token>
   ```

## Creating the First Admin User

Since signup requires admin authentication, you need to create the first admin manually:

```python
from auth import hash_password
from models import User

# Hash the password
hashed = hash_password("your_admin_password")

# Insert manually in MySQL
# INSERT INTO users (name, email, password, role) 
# VALUES ('Admin', 'admin@example.com', '<hashed_password>', 'admin');
```

Or run this script:
```python
python -c "from auth import hash_password; print(hash_password('your_password'))"
```

Then insert the hashed password into the database.

## Error Handling

The API returns appropriate HTTP status codes:
- `200`: Success
- `201`: Created (successful signup)
- `400`: Bad Request (email already exists)
- `401`: Unauthorized (invalid credentials)
- `403`: Forbidden (insufficient permissions)
- `422`: Validation Error (invalid request data)

## Notes

- Change `SECRET_KEY` in `auth.py` to a secure random key in production
- Update CORS settings in `main.py` for production (restrict origins)
- Consider using environment variables for sensitive configuration
- The `image_url` field accepts URLs from your frontend upload system
