from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import router as auth_router
from database import init_database

# Initialize FastAPI app
app = FastAPI(
    title="Authentication API",
    description="FastAPI Authentication System with MySQL",
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

@app.on_event("startup")
async def startup_event():
    """Initialize database on startup"""
    init_database()
    print("Database initialized successfully")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Authentication API",
        "version": "1.0.0",
        "endpoints": {
            "signup": "/auth/signup (POST) - Admin only",
            "login": "/auth/login (POST)",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
