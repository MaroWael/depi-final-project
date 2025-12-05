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
    """
    Run ONNX Runtime inference on a BGR frame and return detections.
    Each detection: {cls: class_id, score: confidence}
    Includes proper Letterboxing, Class-Aware NMS, and Coordinate Mapping.
    """
    if CLEANING_SESSION is None:
        return []
    
    try:
        # Get original frame dimensions
        orig_h, orig_w = frame_bgr.shape[:2]
        
        # -----------------------------
        # 1. Letterbox Preprocessing
        # -----------------------------
        # Calculate scale factor (min scale to fit)
        scale = min(640 / orig_w, 640 / orig_h)
        
        # New dimensions
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        
        # Resize image
        img_resized = cv2.resize(frame_bgr, (new_w, new_h))
        
        # Create canvas (640x640) with gray padding (114)
        canvas = np.full((640, 640, 3), 114, dtype=np.uint8)
        
        # Calculate padding offsets (center the image)
        dw = (640 - new_w) // 2
        dh = (640 - new_h) // 2
        
        # Place resized image on canvas
        canvas[dh:dh+new_h, dw:dw+new_w] = img_resized
        
        # -----------------------------
        # 2. Prepare for Model
        # -----------------------------
        img = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        
        # Run inference
        outputs = CLEANING_SESSION.run(None, {CLEANING_INPUT: img})
        
        # Parse YOLOv8 output: [1, 8, 8400] -> [8400, 8]
        output = outputs[0]
        if len(output.shape) == 3:
            output = output[0].T  # [8, 8400] -> [8400, 8]
        
        # -----------------------------
        # 3. Process Detections
        # -----------------------------
        boxes = []
        scores = []
        class_ids = []
        
        for detection in output:
            # Extract class scores (skip first 4 bbox coordinates)
            class_scores = detection[4:]
            cls_id = int(np.argmax(class_scores))
            confidence = float(class_scores[cls_id])
            
            # Filter by confidence threshold
            if confidence > CONFIDENCE_THRESHOLD:
                # Get bbox coordinates (relative to 640x640 canvas)
                x_center, y_center, width, height = detection[:4]
                
                # Convert to top-left corner format (x, y, w, h) for NMS
                x = x_center - width / 2
                y = y_center - height / 2
                
                boxes.append([x, y, width, height])
                scores.append(confidence)
                class_ids.append(cls_id)
        
        # -----------------------------
        # 4. Class-Aware NMS
        # -----------------------------
        # Offset boxes by class_id * max_wh so NMS is applied per-class
        nms_boxes = []
        max_wh = 4096 # larger than 640
        
        for i, box in enumerate(boxes):
            x, y, w, h = box
            c = class_ids[i]
            nms_boxes.append([x + c * max_wh, y + c * max_wh, w, h])
            
        # Apply NMS
        indices = cv2.dnn.NMSBoxes(
            nms_boxes, 
            scores, 
            score_threshold=CONFIDENCE_THRESHOLD,
            nms_threshold=0.7  # Ultralytics default IoU
        )
        
        detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                # Retrieve original box (from canvas coords)
                x, y, w, h = boxes[i]
                
                # -----------------------------
                # 5. Coordinate Mapping (Canvas -> Original)
                # -----------------------------
                # Remove padding
                x -= dw
                y -= dh
                
                # Scale back to original size
                x /= scale
                y /= scale
                w /= scale
                h /= scale
                
                # Convert to x1, y1, x2, y2
                x1 = int(x)
                y1 = int(y)
                x2 = int(x + w)
                y2 = int(y + h)
                
                # Clip to image bounds
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(orig_w, x2)
                y2 = min(orig_h, y2)
                
                detections.append({
                    "cls": class_ids[i],
                    "score": scores[i],
                    "bbox": [x1, y1, x2, y2]
                })
        
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

# -----------------------------
# Cleaning Model Configuration
# -----------------------------
CONFIDENCE_THRESHOLD = 0.25
FRAME_SKIP_CLEANING = 1  # process every frame

# Colors for visualization (BGR)
CLASS_COLORS = {
    0: (0, 255, 0),      # clean_surface - Green
    1: (0, 165, 255),    # dirty_surface - Orange
    2: (0, 0, 255),      # Rats - Red
    3: (255, 0, 255)     # Insects - Magenta
}

# -----------------------------
# /analyze-cleaning route (ONNX-based)
# -----------------------------
@router.post("/analyze-cleaning", response_model=VideoReportResponse)
async def analyze_cleaning(
    video: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    """
    Upload a video and analyze cleaning/hygiene issues using ONNX model.
    Detects: clean_surface, dirty_surface, Rats, Insects

    This route uses ONNX Runtime for efficient inference.
    """
    if CLEANING_SESSION is None:
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

            # Run ONNX detection
            try:
                detections = run_cleaning_onnx(frame)

                for det in detections:
                    cls_id = det["cls"]
                    conf = det["score"]
                    
                    if 0 <= cls_id < len(CLEAN_CLASSES):
                        cls_name = CLEAN_CLASSES[cls_id]
                        # Update counts and confidence sums
                        class_counts[cls_name] += 1
                        class_confidences[cls_name] += conf

            except Exception as e:
                print(f"ONNX cleaning model error at frame {frame_count}: {e}")

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
