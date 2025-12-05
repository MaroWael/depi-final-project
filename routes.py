from fastapi import APIRouter, Depends, HTTPException, status, Request, UploadFile, File, Form
from fastapi.responses import FileResponse
from schemas import SignupRequest, LoginRequest, LoginResponse, UserResponse, UpdateUserRequest, VideoReportResponse
from models import User, LoginHistory, VideoReport
from auth import hash_password, verify_password, create_access_token, get_current_admin, get_current_user
from datetime import datetime
import os
import uuid
from pathlib import Path
import cv2
from PIL import Image
from inference_sdk import InferenceHTTPClient
from collections import defaultdict
import json
import onnxruntime as ort
import numpy as np

router = APIRouter(prefix="/auth", tags=["Authentication"])

# Create images directory if it doesn't exist
IMAGES_DIR = Path("images")
IMAGES_DIR.mkdir(exist_ok=True)

# Create videos directory if it doesn't exist
VIDEOS_DIR = Path("videos")
VIDEOS_DIR.mkdir(exist_ok=True)

# -----------------------------
# API & MODEL CONFIG
# -----------------------------
MODEL_ID = "saldjs-eodej/1"
API_KEY = "f50xHu5kMJ54A1ERJdnX"
FRAME_SKIP = 5
RESIZE_DIM = (640, 640)

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=API_KEY
)

# Initialize ONNX Cleaning Model
CLEANING_MODEL_PATH = "cleaning_surface.onnx"
CLEANING_SESSION = None
CLEANING_INPUT = None
CLEAN_CLASSES = ["clean_surface", "dirty_surface", "Rats", "Insects"]

try:
    import os
    if not os.path.exists(CLEANING_MODEL_PATH):
        print(f"❌ ONNX model file not found at: {CLEANING_MODEL_PATH}")
        print(f"Current directory: {os.getcwd()}")
        print(f"Files in current directory: {os.listdir('.')}")
    else:
        print(f"✓ ONNX model file found at: {CLEANING_MODEL_PATH}")
        file_size = os.path.getsize(CLEANING_MODEL_PATH)
        print(f"  Model file size: {file_size / (1024*1024):.2f} MB")
        
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        CLEANING_SESSION = ort.InferenceSession(
            CLEANING_MODEL_PATH, 
            sess_options=session_options,
            providers=["CPUExecutionProvider"]
        )
        CLEANING_INPUT = CLEANING_SESSION.get_inputs()[0].name
        
        # Print model info
        input_shape = CLEANING_SESSION.get_inputs()[0].shape
        output_shape = CLEANING_SESSION.get_outputs()[0].shape
        print(f"✓ ONNX Cleaning Model loaded successfully")
        print(f"  Input name: {CLEANING_INPUT}, shape: {input_shape}")
        print(f"  Output shape: {output_shape}")
except Exception as e:
    print(f"❌ Warning: Failed to load ONNX cleaning model: {e}")
    print("The application will continue without cleaning surface detection.")
    import traceback
    traceback.print_exc()
    CLEANING_SESSION = None

def run_cleaning_onnx(frame_bgr):
    try:
        # Get original frame dimensions for scaling boxes
        orig_h, orig_w = frame_bgr.shape[:2]
        
        # Preprocess frame
        img = cv2.resize(frame_bgr, (640, 640))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        
        # Run inference
        outputs = CLEANING_SESSION.run(None, {CLEANING_INPUT: img})
        
        # Parse output - YOLOv8 format
        output = outputs[0]
        
        # Transpose to (8400, 8) if needed
        if len(output.shape) == 3:
            output = output[0].T  # (8, 8400) -> (8400, 8)
        
        detections = []
        
        # Scale factors for bounding boxes
        scale_x = orig_w / 640
        scale_y = orig_h / 640
        
        for detection in output:
            # First 4 values are box coordinates (x_center, y_center, width, height)
            # Remaining values are class confidences
            x_center, y_center, width, height = detection[:4]
            class_scores = detection[4:]
            
            # Get the class with max confidence
            cls_id = int(np.argmax(class_scores))
            confidence = float(class_scores[cls_id])
            
            # Filter by confidence threshold
            if confidence > 0.25:
                detections.append({"cls": cls_id, "score": confidence})
        
        return detections
    except Exception as e:
        print(f"[ONNX] Error in run_cleaning_onnx: {e}")
        import traceback
        traceback.print_exc()
        return []

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
    
    if not image_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Invalid image path"
        )

    return FileResponse(image_path)

@router.post("/analyze-behavior", response_model=VideoReportResponse)
async def analyze_behavior(
    video: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    """
    Upload a video and analyze worker behavior using Roboflow model.
    Detects: eating, face_touching, smoking
    """

    # -----------------------------
    # 1. Save Uploaded Video
    # -----------------------------
    allowed_extensions = {".mp4", ".avi", ".mov", ".mkv"}
    file_ext = os.path.splitext(video.filename)[1].lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail="Invalid video format")

    unique_filename = f"{uuid.uuid4()}{file_ext}"
    video_path = VIDEOS_DIR / unique_filename

    try:
        with open(video_path, "wb") as buffer:
            buffer.write(await video.read())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save video: {e}")

    # -----------------------------
    # 2. Run Behavior Analysis
    # -----------------------------
    cap = cv2.VideoCapture(str(video_path))
    frame_count = 0
    
    # Initialize stats for specific behaviors
    target_behaviors = ["eating", "face_touching", "smoking"]
    behavior_stats = {behavior: {"count": 0, "confidence_sum": 0.0} for behavior in target_behaviors}
    
    # Use defaultdict for any other unexpected behaviors
    other_stats = defaultdict(lambda: {"count": 0, "confidence_sum": 0.0})

    def process_detection(label, conf):
        if not label:
            return
            
        normalized_label = label.lower().replace("-", "_").replace(" ", "_")
        
        # Check behaviors
        if normalized_label in behavior_stats:
            behavior_stats[normalized_label]["count"] += 1
            behavior_stats[normalized_label]["confidence_sum"] += conf
        else:
            # Store other detections
            other_stats[normalized_label]["count"] += 1
            other_stats[normalized_label]["confidence_sum"] += conf

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Skip frames for performance
            if frame_count % FRAME_SKIP != 0:
                continue

            # Resize for model
            frame_resized = cv2.resize(frame, RESIZE_DIM)
            frame_pil = Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))

            # Behavior Detection (Roboflow)
            try:
                result = CLIENT.infer(frame_pil, model_id=MODEL_ID)
                predictions = result.get("predictions", [])
                for pred in predictions:
                    process_detection(pred.get("class"), pred.get("confidence", 0.0))
            except Exception as e:
                print(f"Error in behavior model at frame {frame_count}: {e}")

    finally:
        cap.release()

    # -----------------------------
    # 3. Prepare Final Stats
    # -----------------------------
    final_stats = {}
    
    # Helper to format stats
    def format_stats(stats_dict):
        for key, data in stats_dict.items():
            count = data["count"]
            avg_conf = round(data["confidence_sum"] / count, 3) if count > 0 else 0.0
            is_present = count >= 1
            
            final_stats[key] = {
                "count": count,
                "average_confidence": avg_conf,
                "detected": is_present
            }

    format_stats(behavior_stats)
    format_stats(other_stats)

    final_stats["total_frames_processed"] = frame_count

    # -----------------------------
    # 4. Save to DB
    # -----------------------------
    report_json = json.dumps(final_stats)
    report_id = VideoReport.create(unique_filename, report_json)
    report = VideoReport.get_by_id(report_id)

    # If DB stored string → convert to dict
    report_data = report["report_data"]
    if isinstance(report_data, str):
        report_data = json.loads(report_data)

    return {
        "id": report["id"],
        "video_filename": report["video_filename"],
        "report_data": report_data,
        "created_at": report["created_at"]
    }

from fastapi import APIRouter, Depends, HTTPException, status, Request, UploadFile, File, Form
from fastapi.responses import FileResponse
from schemas import SignupRequest, LoginRequest, LoginResponse, UserResponse, UpdateUserRequest, VideoReportResponse
from models import User, LoginHistory, VideoReport
from auth import hash_password, verify_password, create_access_token, get_current_admin, get_current_user
from datetime import datetime
import os
import uuid
from pathlib import Path
import cv2
from PIL import Image
from inference_sdk import InferenceHTTPClient
from collections import defaultdict
import json
import onnxruntime as ort
import numpy as np

# NEW: ultralytics YOLO import
from ultralytics import YOLO

router = APIRouter(prefix="/auth", tags=["Authentication"])

# Create images directory if it doesn't exist
IMAGES_DIR = Path("images")
IMAGES_DIR.mkdir(exist_ok=True)

# Create videos directory if it doesn't exist
VIDEOS_DIR = Path("videos")
VIDEOS_DIR.mkdir(exist_ok=True)

# -----------------------------
# API & MODEL CONFIG
# -----------------------------
MODEL_ID = "saldjs-eodej/1"
API_KEY = "f50xHu5kMJ54A1ERJdnX"
FRAME_SKIP = 5
RESIZE_DIM = (640, 640)

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=API_KEY
)

# -----------------------------
# ONNX Cleaning Model (kept as-is)
# -----------------------------
CLEANING_MODEL_PATH = "cleaning_surface.onnx"
CLEANING_SESSION = None
CLEANING_INPUT = None
CLEAN_CLASSES = ["clean_surface", "dirty_surface", "Rats", "Insects"]

try:
    import os
    if not os.path.exists(CLEANING_MODEL_PATH):
        print(f"❌ ONNX model file not found at: {CLEANING_MODEL_PATH}")
        print(f"Current directory: {os.getcwd()}")
        print(f"Files in current directory: {os.listdir('.')}")
    else:
        print(f"✓ ONNX model file found at: {CLEANING_MODEL_PATH}")
        file_size = os.path.getsize(CLEANING_MODEL_PATH)
        print(f"  Model file size: {file_size / (1024*1024):.2f} MB")
        
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        CLEANING_SESSION = ort.InferenceSession(
            CLEANING_MODEL_PATH, 
            sess_options=session_options,
            providers=["CPUExecutionProvider"]
        )
        CLEANING_INPUT = CLEANING_SESSION.get_inputs()[0].name
        
        # Print model info
        input_shape = CLEANING_SESSION.get_inputs()[0].shape
        output_shape = CLEANING_SESSION.get_outputs()[0].shape
        print(f"✓ ONNX Cleaning Model loaded successfully")
        print(f"  Input name: {CLEANING_INPUT}, shape: {input_shape}")
        print(f"  Output shape: {output_shape}")
except Exception as e:
    print(f"❌ Warning: Failed to load ONNX cleaning model: {e}")
    print("The application will continue without cleaning surface detection.")
    import traceback
    traceback.print_exc()
    CLEANING_SESSION = None

def run_cleaning_onnx(frame_bgr):
    try:
        # Get original frame dimensions for scaling boxes
        orig_h, orig_w = frame_bgr.shape[:2]
        
        # Preprocess frame
        img = cv2.resize(frame_bgr, (640, 640))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        
        # Run inference
        outputs = CLEANING_SESSION.run(None, {CLEANING_INPUT: img})
        
        # Parse output - YOLOv8 format
        output = outputs[0]
        
        # Transpose to (8400, 8) if needed
        if len(output.shape) == 3:
            output = output[0].T  # (8, 8400) -> (8400, 8)
        
        detections = []
        
        # Scale factors for bounding boxes
        scale_x = orig_w / 640
        scale_y = orig_h / 640
        
        for detection in output:
            # First 4 values are box coordinates (x_center, y_center, width, height)
            # Remaining values are class confidences
            x_center, y_center, width, height = detection[:4]
            class_scores = detection[4:]
            
            # Get the class with max confidence
            cls_id = int(np.argmax(class_scores))
            confidence = float(class_scores[cls_id])
            
            # Filter by confidence threshold
            if confidence > 0.25:
                detections.append({"cls": cls_id, "score": confidence})
        
        return detections
    except Exception as e:
        print(f"[ONNX] Error in run_cleaning_onnx: {e}")
        import traceback
        traceback.print_exc()
        return []

# -----------------------------
# NEW: YOLO Cleaning Model (Option A: add YOLO alongside ONNX)
# -----------------------------
CLEAN_MODEL_PATH = "cleaning_surface.onnx"  # change if needed
CONFIDENCE_THRESHOLD = 0.25
FRAME_SKIP_CLEANING = 1  # process every frame (same as your script)

# Colors for visualization (BGR)
CLASS_COLORS = {
    0: (0, 255, 0),      # clean_surface - Green
    1: (0, 165, 255),    # dirty_surface - Orange
    2: (0, 0, 255),      # Rats - Red
    3: (255, 0, 255)     # Insects - Magenta
}

try:
    print("Loading Cleaning YOLO model (ultralytics)...")
    CLEAN_MODEL = YOLO(CLEAN_MODEL_PATH)
    print("✓ Cleaning YOLO model loaded successfully")
except Exception as e:
    print(f"❌ Error loading YOLO model at {CLEAN_MODEL_PATH}: {e}")
    CLEAN_MODEL = None

def run_cleaning_yolo(frame_bgr):
    """
    Run ultralytics YOLO model on a BGR frame and return detections.
    Each detection: {class_id, class_name, confidence, bbox}
    """
    if CLEAN_MODEL is None:
        return []

    try:
        # ultralytics accepts numpy images (BGR/RGB) — pass frame directly
        results = CLEAN_MODEL(frame_bgr, conf=CONFIDENCE_THRESHOLD, verbose=False)
        detections = []

        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue

            for box in boxes:
                # ultralytics box attributes: cls, conf, xyxy
                try:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    if 0 <= cls_id < len(CLEAN_CLASSES):
                        detections.append({
                            "class_id": cls_id,
                            "class_name": CLEAN_CLASSES[cls_id],
                            "confidence": conf,
                            "bbox": (int(x1), int(y1), int(x2), int(y2))
                        })
                except Exception:
                    # Skip malformed box
                    continue

        return detections

    except Exception as e:
        print(f"[YOLO] Error in run_cleaning_yolo: {e}")
        import traceback
        traceback.print_exc()
        return []

def draw_detections(frame, detections):
    """Optional: draw bounding boxes and labels on frame (BGR image)."""
    for det in detections:
        cls_id = det["class_id"]
        cls_name = det["class_name"]
        conf = det["confidence"]
        x1, y1, x2, y2 = det["bbox"]

        color = CLASS_COLORS.get(cls_id, (255, 255, 255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"{cls_name}: {conf:.2f}"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    return frame

# ----------------------------- existing routes (signup, login, etc.) mostly unchanged
# Keep your original signup/login/get_all_chiefs/update_user/delete_user/get_user_image
# and analyze_behavior route exactly as in your original file.
# For brevity I include them unchanged below.
# -----------------------------

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
    
    if not image_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Invalid image path"
        )

    return FileResponse(image_path)

@router.post("/analyze-behavior", response_model=VideoReportResponse)
async def analyze_behavior(
    video: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    """
    Upload a video and analyze worker behavior using Roboflow model.
    Detects: eating, face_touching, smoking
    """

    # -----------------------------
    # 1. Save Uploaded Video
    # -----------------------------
    allowed_extensions = {".mp4", ".avi", ".mov", ".mkv"}
    file_ext = os.path.splitext(video.filename)[1].lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail="Invalid video format")

    unique_filename = f"{uuid.uuid4()}{file_ext}"
    video_path = VIDEOS_DIR / unique_filename

    try:
        with open(video_path, "wb") as buffer:
            buffer.write(await video.read())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save video: {e}")

    # -----------------------------
    # 2. Run Behavior Analysis
    # -----------------------------
    cap = cv2.VideoCapture(str(video_path))
    frame_count = 0
    
    # Initialize stats for specific behaviors
    target_behaviors = ["eating", "face_touching", "smoking"]
    behavior_stats = {behavior: {"count": 0, "confidence_sum": 0.0} for behavior in target_behaviors}
    
    # Use defaultdict for any other unexpected behaviors
    other_stats = defaultdict(lambda: {"count": 0, "confidence_sum": 0.0})

    def process_detection(label, conf):
        if not label:
            return
            
        normalized_label = label.lower().replace("-", "_").replace(" ", "_")
        
        # Check behaviors
        if normalized_label in behavior_stats:
            behavior_stats[normalized_label]["count"] += 1
            behavior_stats[normalized_label]["confidence_sum"] += conf
        else:
            # Store other detections
            other_stats[normalized_label]["count"] += 1
            other_stats[normalized_label]["confidence_sum"] += conf

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Skip frames for performance
            if frame_count % FRAME_SKIP != 0:
                continue

            # Resize for model
            frame_resized = cv2.resize(frame, RESIZE_DIM)
            frame_pil = Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))

            # Behavior Detection (Roboflow)
            try:
                result = CLIENT.infer(frame_pil, model_id=MODEL_ID)
                predictions = result.get("predictions", [])
                for pred in predictions:
                    process_detection(pred.get("class"), pred.get("confidence", 0.0))
            except Exception as e:
                print(f"Error in behavior model at frame {frame_count}: {e}")

    finally:
        cap.release()

    # -----------------------------
    # 3. Prepare Final Stats
    # -----------------------------
    final_stats = {}
    
    # Helper to format stats
    def format_stats(stats_dict):
        for key, data in stats_dict.items():
            count = data["count"]
            avg_conf = round(data["confidence_sum"] / count, 3) if count > 0 else 0.0
            is_present = count >= 1
            
            final_stats[key] = {
                "count": count,
                "average_confidence": avg_conf,
                "detected": is_present
            }

    format_stats(behavior_stats)
    format_stats(other_stats)

    final_stats["total_frames_processed"] = frame_count

    # -----------------------------
    # 4. Save to DB
    # -----------------------------
    report_json = json.dumps(final_stats)
    report_id = VideoReport.create(unique_filename, report_json)
    report = VideoReport.get_by_id(report_id)

    # If DB stored string → convert to dict
    report_data = report["report_data"]
    if isinstance(report_data, str):
        report_data = json.loads(report_data)

    return {
        "id": report["id"],
        "video_filename": report["video_filename"],
        "report_data": report_data,
        "created_at": report["created_at"]
    }

# -----------------------------
# REPLACED: /analyze-cleaning route (YOLO-based) - Option A
# -----------------------------
@router.post("/analyze-cleaning", response_model=VideoReportResponse)
async def analyze_cleaning(
    video: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    """
    Upload a video and analyze cleaning/hygiene issues using YOLO model.
    Detects: clean_surface, dirty_surface, Rats, Insects

    This route uses the ultralytics YOLO model if available (CLEAN_MODEL).
    If CLEAN_MODEL is not loaded, it will return 503.
    """
    if CLEAN_MODEL is None:
        # If user wants ONNX fallback, they can re-enable run_cleaning_onnx logic.
        raise HTTPException(
            status_code=503,
            detail="Cleaning detection model is not available"
        )

    # -----------------------------
    # 1. Save Uploaded Video
    # -----------------------------
    allowed_extensions = {".mp4", ".avi", ".mov", ".mkv"}
    file_ext = os.path.splitext(video.filename)[1].lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail="Invalid video format")

    unique_filename = f"{uuid.uuid4()}{file_ext}"
    video_path = VIDEOS_DIR / unique_filename

    try:
        with open(video_path, "wb") as buffer:
            buffer.write(await video.read())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save video: {e}")

    # -----------------------------
    # 2. Run Cleaning Analysis (YOLO)
    # -----------------------------
    cap = cv2.VideoCapture(str(video_path))
    frame_count = 0

    # Stats per class name (keys: CLEAN_CLASSES)
    class_counts = defaultdict(int)
    class_confidences = defaultdict(float)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Skip frames according to FRAME_SKIP_CLEANING
            if frame_count % FRAME_SKIP_CLEANING != 0:
                continue

            # Optionally: resize to a reasonable size for speed (YOLO will handle scale)
            # frame_for_model = cv2.resize(frame, RESIZE_DIM)
            frame_for_model = frame

            # Run YOLO detection
            try:
                detections = run_cleaning_yolo(frame_for_model)

                for det in detections:
                    cls_name = det["class_name"]
                    conf = det["confidence"]

                    # Update counts and confidence sums
                    class_counts[cls_name] += 1
                    class_confidences[cls_name] += conf

            except Exception as e:
                print(f"YOLO cleaning model error at frame {frame_count}: {e}")

    finally:
        cap.release()

    # -----------------------------
    # 3. Prepare Final Stats
    # -----------------------------
    final_stats = {}

    # Ensure we always output the same keys as before
    # Rats
    rats_count = class_counts["Rats"]
    rats_conf = class_confidences["Rats"]
    final_stats["rats"] = {
        "count": rats_count,
        "average_confidence": round(rats_conf / rats_count, 3) if rats_count > 0 else 0.0,
        "detected": rats_count > 0
    }

    # Insects
    insects_count = class_counts["Insects"]
    insects_conf = class_confidences["Insects"]
    final_stats["insects"] = {
        "count": insects_count,
        "average_confidence": round(insects_conf / insects_count, 3) if insects_count > 0 else 0.0,
        "detected": insects_count > 0
    }

    # Dirty Surface
    dirty_count = class_counts["dirty_surface"]
    dirty_conf = class_confidences["dirty_surface"]
    final_stats["dirty"] = {
        "count": dirty_count,
        "average_confidence": round(dirty_conf / dirty_count, 3) if dirty_count > 0 else 0.0,
        "detected": dirty_count > 0
    }

    # Clean Surface (optional)
    clean_count = class_counts["clean_surface"]
    clean_conf = class_confidences["clean_surface"]
    final_stats["clean_surface"] = {
        "count": clean_count,
        "average_confidence": round(clean_conf / clean_count, 3) if clean_count > 0 else 0.0,
        "detected": clean_count > 0
    }

    final_stats["total_frames_processed"] = frame_count

    # -----------------------------
    # Determine Cleanliness Status
    # -----------------------------
    has_rats = final_stats["rats"]["detected"]
    has_insects = final_stats["insects"]["detected"]
    is_dirty = final_stats["dirty"]["detected"]

    final_stats["is_clean"] = not (has_rats or has_insects or is_dirty)

    # -----------------------------
    # 4. Save to DB
    # -----------------------------
    report_json = json.dumps(final_stats)
    report_id = VideoReport.create(unique_filename, report_json)
    report = VideoReport.get_by_id(report_id)

    # If DB stored string → convert to dict
    report_data = report["report_data"]
    if isinstance(report_data, str):
        report_data = json.loads(report_data)

    return {
        "id": report["id"],
        "video_filename": report["video_filename"],
        "report_data": report_data,
        "created_at": report["created_at"]
    }
