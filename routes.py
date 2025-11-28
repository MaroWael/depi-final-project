from fastapi import APIRouter, Depends, HTTPException, status, Request, UploadFile, File, Form
from fastapi.responses import FileResponse
from schemas import SignupRequest, LoginRequest, LoginResponse, UserResponse
from models import User, LoginHistory
from auth import hash_password, verify_password, create_access_token, get_current_admin
from datetime import datetime
import os
import uuid
from pathlib import Path

router = APIRouter(prefix="/auth", tags=["Authentication"])

# Create images directory if it doesn't exist
IMAGES_DIR = Path("images")
IMAGES_DIR.mkdir(exist_ok=True)

@router.post("/signup", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def signup(
    name: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    role: str = Form(...),
    image: UploadFile = File(...),
    current_admin: dict = Depends(get_current_admin)
):
    """
    Register a new user (admin or chief).
    Only admins can register new users.
    Accepts multipart form data with image upload.
    """
    # Validate role
    if role not in ["admin", "chief"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Role must be either 'admin' or 'chief'"
        )
    
    # Validate password length
    if len(password) < 6:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must be at least 6 characters long"
        )
    
    # Check if email already exists
    if User.email_exists(email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Validate image file
    allowed_extensions = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
    file_ext = os.path.splitext(image.filename)[1].lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid image format. Allowed formats: {', '.join(allowed_extensions)}"
        )
    
    # Generate unique filename
    unique_filename = f"{uuid.uuid4()}{file_ext}"
    image_path = IMAGES_DIR / unique_filename
    
    # Save image file
    try:
        with open(image_path, "wb") as buffer:
            content = await image.read()
            buffer.write(content)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save image: {str(e)}"
        )
    
    # Hash the password
    hashed_password = hash_password(password)
    
    # Create the user with image filename
    try:
        user_id = User.create(
            name=name,
            email=email,
            password=hashed_password,
            image_url=unique_filename,  # Store only filename, not full path
            role=role
        )
    except Exception as e:
        # Delete uploaded image if user creation fails
        if image_path.exists():
            os.remove(image_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create user: {str(e)}"
        )
    
    # Get the created user
    user = User.get_by_id(user_id)
    
    return UserResponse(
        id=user["id"],
        name=user["name"],
        email=user["email"],
        image_url=user["image_url"],
        role=user["role"],
        created_at=user["created_at"]
    )

@router.post("/login", response_model=LoginResponse)
async def login(login_data: LoginRequest, request: Request):
    """
    Login endpoint - authenticates user and creates login history record.
    """
    # Get user by email
    user = User.get_by_email(login_data.email)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )
    
    # Verify password
    if not verify_password(login_data.password, user["password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )
    
    # Create access token
    access_token = create_access_token(
        data={
            "user_id": user["id"],
            "email": user["email"],
            "role": user["role"]
        }
    )
    
    # Record login history
    client_ip = request.client.host if request.client else None
    user_agent = request.headers.get("user-agent")
    LoginHistory.create(
        user_id=user["id"],
        ip_address=client_ip,
        user_agent=user_agent
    )
    
    # Return response
    return LoginResponse(
        access_token=access_token,
        user=UserResponse(
            id=user["id"],
            name=user["name"],
            email=user["email"],
            image_url=user["image_url"],
            role=user["role"],
            created_at=user["created_at"]
        )
    )

@router.get("/images/{filename}")
async def get_image(filename: str):
    """
    Serve uploaded user images
    """
    image_path = IMAGES_DIR / filename
    
    if not image_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Image not found"
        )
    
    return FileResponse(image_path)
