from fastapi import APIRouter, Depends, HTTPException, status, Request, UploadFile, File, Form
from fastapi.responses import FileResponse
from schemas import SignupRequest, LoginRequest, LoginResponse, UserResponse, UpdateUserRequest
from models import User, LoginHistory
from auth import hash_password, verify_password, create_access_token, get_current_admin, get_current_user
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


@router.get("/chiefs")
async def get_all_chiefs(current_admin: dict = Depends(get_current_admin)):
    """
    Get all chiefs (Admin only).
    Returns a list of all users with the 'chief' role.
    Non-admin users will receive an access denied error.
    """
    chiefs = User.get_all_chiefs()
    
    return {
        "chiefs": [
            UserResponse(
                id=chief["id"],
                name=chief["name"],
                email=chief["email"],
                image_url=chief["image_url"],
                role=chief["role"],
                created_at=chief["created_at"]
            )
            for chief in chiefs
        ],
        "total": len(chiefs)
    }

@router.put("/user/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: int,
    name: str = Form(None),
    email: str = Form(None),
    password: str = Form(None),
    role: str = Form(None),
    image: UploadFile = File(None),
    current_admin: dict = Depends(get_current_admin)
):
    """
    Update user information (Admin only).
    All fields are optional. Only provided fields will be updated.
    """
    # Check if user exists
    user = User.get_by_id(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Validate role if provided
    if role and role not in ["admin", "chief"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Role must be either 'admin' or 'chief'"
        )
    
    # Validate password length if provided
    if password and len(password) < 6:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must be at least 6 characters long"
        )
    
    # Check if email is being changed and if it already exists
    if email and email != user["email"]:
        if User.email_exists_excluding_user(email, user_id):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already in use by another user"
            )
    
    # Handle image upload if provided
    new_image_url = None
    if image:
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
        
        # Save new image file
        try:
            with open(image_path, "wb") as buffer:
                content = await image.read()
                buffer.write(content)
            new_image_url = unique_filename
            
            # Delete old image if exists
            if user["image_url"]:
                old_image_path = IMAGES_DIR / user["image_url"]
                if old_image_path.exists():
                    os.remove(old_image_path)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to save image: {str(e)}"
            )
    
    # Hash password if provided
    hashed_password = hash_password(password) if password else None
    
    # Update user
    try:
        User.update(
            user_id=user_id,
            name=name,
            email=email,
            password=hashed_password,
            image_url=new_image_url,
            role=role
        )
    except Exception as e:
        # Delete new image if update fails
        if new_image_url:
            new_image_path = IMAGES_DIR / new_image_url
            if new_image_path.exists():
                os.remove(new_image_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update user: {str(e)}"
        )
    
    # Get updated user
    updated_user = User.get_by_id(user_id)
    
    return UserResponse(
        id=updated_user["id"],
        name=updated_user["name"],
        email=updated_user["email"],
        image_url=updated_user["image_url"],
        role=updated_user["role"],
        created_at=updated_user["created_at"]
    )

@router.delete("/user/{user_id}")
async def delete_user(
    user_id: int,
    current_admin: dict = Depends(get_current_admin)
):
    """
    Delete a user (Admin only).
    This will also delete the user's image and login history (cascade).
    """
    # Check if user exists
    user = User.get_by_id(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Prevent admin from deleting themselves
    if user_id == current_admin["id"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You cannot delete your own account"
        )
    
    # Delete user's image if exists
    if user["image_url"]:
        image_path = IMAGES_DIR / user["image_url"]
        if image_path.exists():
            try:
                os.remove(image_path)
            except Exception as e:
                print(f"Warning: Failed to delete image: {e}")
    
    # Delete user from database
    try:
        User.delete(user_id)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete user: {str(e)}"
        )
    
    return {
        "message": "User deleted successfully",
        "user_id": user_id,
        "email": user["email"]
    }

@router.get("/images/{filename}")
async def get_user_image(filename: str):
    """
    Serve user profile images.
    Public endpoint - no authentication required.
    """
    image_path = IMAGES_DIR / filename
    
    if not image_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Image not found"
        )
    
    # Check if it's actually a file (security)
    if not image_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Invalid image path"
        )
    
    return FileResponse(image_path)
