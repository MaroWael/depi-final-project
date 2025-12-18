from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import router as auth_router, get_cleaning_session, get_ppe_session
from database import init_database
import threading
import time

# Initialize FastAPI app
app = FastAPI(
    title="Safety & Hygiene Monitoring API",
    description="API for Video Analysis, PPE Detection, and Hygiene Monitoring",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth_router)

@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Welcome to the Safety & Hygiene Monitoring API",
        "docs": "/docs",
        "endpoints": {
            "signup": "/signup",
            "login": "/login",
            "analyze_behavior": "/analyze-behavior",
            "analyze_cleaning": "/analyze-cleaning",
            "analyze_ppe": "/analyze-ppe"
        }
    }

@app.on_event("startup")
async def startup_event():
    """Initialize database and load models on startup"""
    print("--- Startup Event Begin ---")
    init_database()
    print("Database initialized successfully")

    def load_models_background():
        print("[Background] Starting model loading...")
        start_time = time.time()
        
        try:
            print("[Background] Loading Cleaning Model...")
            get_cleaning_session()
            print("[Background] Cleaning Model loaded.")
        except Exception as e:
            print(f"[Background] ❌ Error loading Cleaning Model: {e}")

        try:
            print("[Background] Loading PPE Model...")
            get_ppe_session()
            print("[Background] PPE Model loaded.")
        except Exception as e:
            print(f"[Background] ❌ Error loading PPE Model: {e}")
            
        elapsed = time.time() - start_time
        print(f"[Background] Model loading completed in {elapsed:.2f} seconds.")

    # Start background thread for model loading
    thread = threading.Thread(target=load_models_background, daemon=True)
    thread.start()
    print("--- Startup Event End (Background thread started) ---")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
