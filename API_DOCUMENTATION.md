# API Documentation

## Base URL
```
http://localhost:8000
```

## Authentication
Most endpoints require a Bearer token in the Authorization header:
```
Authorization: Bearer <your_jwt_token>
```

---

## Endpoints

### 1. Root Endpoint
Get API information.

**Endpoint:** `GET /`

**Authentication:** Not required

**Response:**
```json
{
  "message": "Authentication API",
  "version": "1.0.0",
  "endpoints": {
    "signup": "/auth/signup (POST) - Admin only",
    "login": "/auth/login (POST)",
    "docs": "/docs"
  }
}
```

---

### 2. Health Check
Check if the API is running.

**Endpoint:** `GET /health`

**Authentication:** Not required

**Response:**
```json
{
  "status": "healthy"
}
```

---

### 3. User Login
Authenticate a user and receive an access token.

**Endpoint:** `POST /auth/login`

**Authentication:** Not required

**Request Body:**
```json
{
  "email": "user@example.com",
  "password": "password123"
}
```

**Success Response (200 OK):**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoxLCJlbWFpbCI6InVzZXJAZXhhbXBsZS5jb20iLCJyb2xlIjoiYWRtaW4iLCJleHAiOjE3MzI4ODY0MDB9.abc123...",
  "token_type": "bearer",
  "user": {
    "id": 1,
    "name": "John Doe",
    "email": "user@example.com",
    "image_url": "https://example.com/photo.jpg",
    "role": "admin",
    "created_at": "2025-11-28T10:30:00"
  }
}
```

**Error Response (401 Unauthorized):**
```json
{
  "detail": "Invalid email or password"
}
```

**Side Effects:**
- Creates a record in `login_history` table with:
  - User ID
  - Login timestamp
  - IP address
  - User agent

---

### 4. User Signup (Admin Only)
Register a new user (admin or chief role).

**Endpoint:** `POST /auth/signup`

**Authentication:** Required (Admin only)

**Headers:**
```
Authorization: Bearer <admin_token>
Content-Type: application/json
```

**Request Body:**
```json
{
  "name": "Jane Smith",
  "email": "jane@example.com",
  "password": "securepass123",
  "image_url": "https://example.com/jane.jpg",
  "role": "chief"
}
```

**Field Validation:**
- `name`: Required, 1-255 characters
- `email`: Required, valid email format, must be unique
- `password`: Required, minimum 6 characters
- `image_url`: Optional, URL string
- `role`: Required, either "admin" or "chief"

**Success Response (201 Created):**
```json
{
  "id": 2,
  "name": "Jane Smith",
  "email": "jane@example.com",
  "image_url": "https://example.com/jane.jpg",
  "role": "chief",
  "created_at": "2025-11-28T11:00:00"
}
```

**Error Response (400 Bad Request):**
```json
{
  "detail": "Email already registered"
}
```

**Error Response (401 Unauthorized):**
```json
{
  "detail": "Invalid authentication credentials"
}
```

**Error Response (403 Forbidden):**
```json
{
  "detail": "Only admins can perform this action"
}
```

**Error Response (422 Validation Error):**
```json
{
  "detail": [
    {
      "loc": ["body", "email"],
      "msg": "value is not a valid email address",
      "type": "value_error.email"
    }
  ]
}
```

---

## Data Models

### User
```typescript
{
  id: number;              // Auto-generated
  name: string;            // User's full name
  email: string;           // Unique email address
  password: string;        // Hashed password (never returned in responses)
  image_url: string | null; // URL to profile photo
  role: "admin" | "chief"; // User role
  created_at: datetime;    // Account creation timestamp
}
```

### Login History
```typescript
{
  id: number;          // Auto-generated
  user_id: number;     // Foreign key to users table
  login_time: datetime; // Timestamp of login
  ip_address: string;  // Client IP address
  user_agent: string;  // Browser/client user agent
}
```

---

## Error Codes

| Code | Meaning | Description |
|------|---------|-------------|
| 200 | OK | Request successful |
| 201 | Created | Resource created successfully |
| 400 | Bad Request | Invalid request data or duplicate email |
| 401 | Unauthorized | Invalid or missing authentication token |
| 403 | Forbidden | Insufficient permissions (not an admin) |
| 422 | Validation Error | Request data doesn't match schema |
| 500 | Internal Server Error | Server error |

---

## Authentication Flow

### Initial Setup
1. Create database: `CREATE DATABASE ai_db;`
2. Run `python database.py` to create tables
3. Run `python create_admin.py` to create first admin

### Normal Usage Flow
1. **Admin logs in** → Receives JWT token
2. **Admin registers new users** → Using the token
3. **New users log in** → Receive their own tokens
4. **All logins tracked** → Stored in `login_history` table

---

## Example Requests

### Using cURL

**Login:**
```bash
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "admin@example.com",
    "password": "admin123"
  }'
```

**Signup (with token):**
```bash
curl -X POST http://localhost:8000/auth/signup \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer eyJhbGc..." \
  -d '{
    "name": "New User",
    "email": "newuser@example.com",
    "password": "password123",
    "image_url": "https://example.com/photo.jpg",
    "role": "chief"
  }'
```

### Using Python (requests)

```python
import requests

# Login
response = requests.post(
    "http://localhost:8000/auth/login",
    json={
        "email": "admin@example.com",
        "password": "admin123"
    }
)
data = response.json()
token = data["access_token"]

# Signup
response = requests.post(
    "http://localhost:8000/auth/signup",
    headers={"Authorization": f"Bearer {token}"},
    json={
        "name": "New User",
        "email": "newuser@example.com",
        "password": "password123",
        "image_url": "https://example.com/photo.jpg",
        "role": "chief"
    }
)
```

### Using JavaScript (fetch)

```javascript
// Login
const loginRes = await fetch('http://localhost:8000/auth/login', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    email: 'admin@example.com',
    password: 'admin123'
  })
});
const { access_token } = await loginRes.json();

// Signup
const signupRes = await fetch('http://localhost:8000/auth/signup', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${access_token}`
  },
  body: JSON.stringify({
    name: 'New User',
    email: 'newuser@example.com',
    password: 'password123',
    image_url: 'https://example.com/photo.jpg',
    role: 'chief'
  })
});
```

---

## Security Best Practices

1. **Change SECRET_KEY** in `auth.py` to a secure random string in production
2. **Use HTTPS** in production to protect tokens in transit
3. **Store tokens securely** on the client (httpOnly cookies recommended)
4. **Validate token expiration** - tokens expire after 24 hours
5. **Update CORS settings** in `main.py` to restrict origins in production
6. **Use environment variables** for sensitive configuration
7. **Implement rate limiting** to prevent brute force attacks
8. **Log security events** from the login_history table

---

## Interactive Documentation

FastAPI provides automatic interactive API documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

These interfaces allow you to:
- View all endpoints
- See request/response schemas
- Test endpoints directly in the browser
- Generate code examples
