# Quick Start Guide

## Step-by-Step Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Ensure MySQL is Running
Make sure MySQL is running on your machine with:
- Username: `root`
- Password: `1234`
- Create database: `ai_db`

```sql
CREATE DATABASE ai_db;
```

### 3. Initialize Database Tables
```bash
python database.py
```

This creates the `users` and `login_history` tables.

### 4. Create First Admin User
```bash
python create_admin.py
```

Follow the prompts to create your first admin account.

### 5. Start the Server
```bash
python main.py
```

The API will be available at `http://localhost:8000`

### 6. Test the API
Open your browser and go to: `http://localhost:8000/docs`

Or run the test script:
```bash
python test_api.py
```

## Quick API Usage

### Login (Get Token)
```bash
curl -X POST "http://localhost:8000/auth/login" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "admin@example.com",
    "password": "your_password"
  }'
```

### Register New User (Admin Only)
```bash
curl -X POST "http://localhost:8000/auth/signup" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN_HERE" \
  -d '{
    "name": "John Doe",
    "email": "john@example.com",
    "password": "secure123",
    "image_url": "https://example.com/photo.jpg",
    "role": "chief"
  }'
```

## Frontend Integration Example

```javascript
// Login
const loginResponse = await fetch('http://localhost:8000/auth/login', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    email: 'user@example.com',
    password: 'password123'
  })
});

const { access_token, user } = await loginResponse.json();
localStorage.setItem('token', access_token);

// Register new user (admin only)
const signupResponse = await fetch('http://localhost:8000/auth/signup', {
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

## Important Notes

1. **First Admin**: You must create the first admin using `create_admin.py` because signup requires admin authentication.

2. **Admin Access**: Only users with `role="admin"` can register new users.

3. **Login Tracking**: Every login is automatically recorded in the `login_history` table with IP address and user agent.

4. **Token Expiration**: JWT tokens expire after 24 hours by default.

5. **Image URL**: The `image_url` field should contain the URL of the uploaded user photo from your frontend.

## Troubleshooting

### MySQL Connection Error
- Verify MySQL is running
- Check credentials in `database.py`
- Ensure `ai_db` database exists

### Import Errors
- Run `pip install -r requirements.txt`
- Use a virtual environment

### Token Errors
- Ensure you're including the token in the Authorization header
- Check if the token has expired (24 hours)
- Verify you're using `Bearer` prefix: `Authorization: Bearer <token>`
