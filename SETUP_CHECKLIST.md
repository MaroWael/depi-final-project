# 📋 Complete Setup Checklist

Use this checklist to set up your FastAPI Authentication System step by step.

---

## ✅ Pre-requisites

- [ ] Python 3.8+ installed
- [ ] MySQL 8.x installed and running
- [ ] pip package manager available
- [ ] Git installed (optional, for version control)
- [ ] Text editor or IDE (VS Code, PyCharm, etc.)

---

## 📦 Step 1: Project Setup

- [ ] Navigate to project directory
  ```powershell
  cd "d:\Self Study\DEPI Final Project"
  ```

- [ ] Create virtual environment (recommended)
  ```powershell
  python -m venv venv
  ```

- [ ] Activate virtual environment
  ```powershell
  .\venv\Scripts\Activate
  ```

- [ ] Install dependencies
  ```powershell
  pip install -r requirements.txt
  ```

---

## 🗄️ Step 2: Database Setup

- [ ] Start MySQL service
  ```powershell
  # Windows: Services or MySQL Workbench
  net start MySQL80
  ```

- [ ] Create database
  ```sql
  mysql -u root -p1234 -e "CREATE DATABASE IF NOT EXISTS ai_db;"
  ```
  Or use MySQL Workbench:
  ```sql
  CREATE DATABASE IF NOT EXISTS ai_db;
  ```

- [ ] Verify database exists
  ```sql
  SHOW DATABASES;
  ```

- [ ] Initialize database tables
  ```powershell
  python database.py
  ```

- [ ] Verify tables created
  ```sql
  USE ai_db;
  SHOW TABLES;
  -- Should show: users, login_history
  ```

- [ ] Check table structures
  ```sql
  DESCRIBE users;
  DESCRIBE login_history;
  ```

---

## 👤 Step 3: Create First Admin User

- [ ] Run admin creation script
  ```powershell
  python create_admin.py
  ```

- [ ] Enter admin details when prompted:
  - Name: `Your Name`
  - Email: `admin@example.com`
  - Password: `admin123` (minimum 6 characters)
  - Image URL: `https://example.com/admin.jpg` (optional)

- [ ] Verify admin created successfully
  ```sql
  SELECT id, name, email, role FROM users;
  ```

---

## 🚀 Step 4: Start the Application

- [ ] Start FastAPI server
  ```powershell
  python main.py
  ```

- [ ] Verify server is running
  - Console should show: "Uvicorn running on http://0.0.0.0:8000"
  - No error messages appear

- [ ] Test server is accessible
  ```powershell
  # In a new PowerShell window
  curl http://localhost:8000
  ```

---

## 🧪 Step 5: Test the API

### Option A: Use Interactive Docs

- [ ] Open browser to http://localhost:8000/docs
- [ ] See Swagger UI with all endpoints
- [ ] Expand `/auth/login` endpoint
- [ ] Click "Try it out"
- [ ] Enter admin credentials:
  ```json
  {
    "email": "admin@example.com",
    "password": "admin123"
  }
  ```
- [ ] Click "Execute"
- [ ] Verify 200 response with token
- [ ] Copy the `access_token` value

### Option B: Use Test Script

- [ ] Open new terminal (keep server running)
- [ ] Run test script
  ```powershell
  python test_api.py
  ```
- [ ] Verify all tests pass

### Option C: Use cURL

- [ ] Test login endpoint
  ```powershell
  curl -X POST "http://localhost:8000/auth/login" `
    -H "Content-Type: application/json" `
    -d '{\"email\":\"admin@example.com\",\"password\":\"admin123\"}'
  ```
- [ ] Verify you receive a token in response

---

## 👥 Step 6: Test User Registration

- [ ] Get admin token from login (Step 5)

- [ ] Test signup endpoint
  ```powershell
  curl -X POST "http://localhost:8000/auth/signup" `
    -H "Content-Type: application/json" `
    -H "Authorization: Bearer YOUR_TOKEN_HERE" `
    -d '{
      \"name\":\"Chef John\",
      \"email\":\"chef@example.com\",
      \"password\":\"chef123456\",
      \"image_url\":\"https://example.com/chef.jpg\",
      \"role\":\"chief\"
    }'
  ```

- [ ] Verify 201 response with new user data

- [ ] Verify new user can login
  ```powershell
  curl -X POST "http://localhost:8000/auth/login" `
    -H "Content-Type: application/json" `
    -d '{\"email\":\"chef@example.com\",\"password\":\"chef123456\"}'
  ```

---

## 📊 Step 7: Verify Database Records

- [ ] Check users table
  ```sql
  SELECT id, name, email, role, created_at FROM users;
  ```
  Should show both admin and chef users

- [ ] Check login history
  ```sql
  SELECT 
    lh.id, 
    u.name, 
    u.email, 
    lh.login_time, 
    lh.ip_address 
  FROM login_history lh
  JOIN users u ON lh.user_id = u.id
  ORDER BY lh.login_time DESC;
  ```
  Should show login records for each successful login

---

## 🔐 Step 8: Security Configuration

- [ ] Change SECRET_KEY in `auth.py`
  ```python
  # Generate a secure key:
  import secrets
  print(secrets.token_urlsafe(32))
  ```
  Replace `SECRET_KEY` value in `auth.py`

- [ ] Update CORS settings in `main.py` (if needed)
  ```python
  allow_origins=["http://localhost:3000"]  # Your frontend URL
  ```

- [ ] Create `.env` file from template
  ```powershell
  Copy-Item .env.example .env
  ```

- [ ] Update `.env` with your settings (optional enhancement)

---

## 📱 Step 9: Frontend Integration

- [ ] Note API base URL: `http://localhost:8000`

- [ ] Document required headers:
  - `Content-Type: application/json`
  - `Authorization: Bearer <token>` (for signup)

- [ ] Test login from frontend
  - [ ] POST to `/auth/login`
  - [ ] Store returned token
  - [ ] Display user info

- [ ] Test signup from frontend (admin only)
  - [ ] POST to `/auth/signup` with admin token
  - [ ] Handle success/error responses

- [ ] Implement image upload flow
  - [ ] Upload image to storage
  - [ ] Get image URL
  - [ ] Include URL in signup request

---

## ✅ Step 10: Final Verification

### API Health
- [ ] Server starts without errors
- [ ] All endpoints respond correctly
- [ ] Interactive docs accessible at `/docs`
- [ ] Root endpoint returns API info

### Database
- [ ] Both tables exist and are populated
- [ ] Foreign key relationship works
- [ ] Login history records created on each login
- [ ] No duplicate emails allowed

### Authentication
- [ ] Admin can login successfully
- [ ] Admin can register new users
- [ ] New users can login
- [ ] Non-admin cannot access signup endpoint
- [ ] Invalid credentials are rejected

### Security
- [ ] Passwords are hashed (not stored in plain text)
- [ ] JWT tokens are generated correctly
- [ ] Tokens expire after 24 hours
- [ ] Role-based access control works

---

## 🐛 Troubleshooting

### MySQL Connection Failed
```
Error: Can't connect to MySQL server
```
**Solution:**
- [ ] Verify MySQL is running: `net start MySQL80`
- [ ] Check credentials in `database.py`
- [ ] Ensure `ai_db` database exists
- [ ] Test connection: `mysql -u root -p1234`

### Import Errors
```
ModuleNotFoundError: No module named 'fastapi'
```
**Solution:**
- [ ] Activate virtual environment
- [ ] Install requirements: `pip install -r requirements.txt`
- [ ] Verify installation: `pip list`

### Port Already in Use
```
Error: Address already in use
```
**Solution:**
- [ ] Change port in `main.py`: `uvicorn.run(..., port=8001)`
- [ ] Or kill process using port 8000:
  ```powershell
  Get-Process -Id (Get-NetTCPConnection -LocalPort 8000).OwningProcess | Stop-Process
  ```

### Token Invalid/Expired
```
401 Unauthorized: Invalid authentication credentials
```
**Solution:**
- [ ] Login again to get a new token
- [ ] Check token format: `Bearer <token>`
- [ ] Verify token not expired (24 hours)

### Email Already Registered
```
400 Bad Request: Email already registered
```
**Solution:**
- [ ] Use a different email
- [ ] Or delete existing user from database:
  ```sql
  DELETE FROM users WHERE email = 'duplicate@example.com';
  ```

---

## 📚 Next Steps

Once everything is working:

- [ ] Read `README.md` for detailed documentation
- [ ] Review `API_DOCUMENTATION.md` for endpoint details
- [ ] Check `ARCHITECTURE.md` to understand system design
- [ ] Explore `QUICKSTART.md` for common operations
- [ ] Review security best practices in docs

---

## 🎉 Success Criteria

You've successfully set up the system when:

✅ Server runs without errors
✅ Can login as admin
✅ Can register new users (admin only)
✅ New users can login
✅ Login history is tracked
✅ API documentation is accessible
✅ Database has correct records
✅ Frontend can integrate successfully

---

## 📞 Need Help?

1. Check error messages in console
2. Review relevant documentation file
3. Verify database connection
4. Check API docs at `/docs`
5. Review troubleshooting section above

---

**Last Updated:** November 28, 2025
**Version:** 1.0.0
