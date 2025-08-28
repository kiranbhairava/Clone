from fastapi import FastAPI, Depends, HTTPException, Query, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.sessions import SessionMiddleware
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import or_
import google.generativeai as genai
from dotenv import load_dotenv
import os
import logging
from werkzeug.security import generate_password_hash, check_password_hash
import jwt
from datetime import datetime, timedelta, timezone
from typing import Optional, Set
from contextlib import asynccontextmanager
from simple_logging import log_chat_interaction, log_user_activity, log_error, log_api_endpoint, app_logger
import time
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from schemas import MobileUpdateRequest
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration
JWT_SECRET = os.getenv("JWT_SECRET")
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_DAYS = int(os.getenv("JWT_EXPIRY_MINUTES", "5"))

# Import modules
from database import get_db, Base, engine, test_db_connection
from models import User, Character, CharacterKnowledge, ConversationSession, ConversationMessage, CHARACTER_CATEGORIES
from schemas import *
from multilingual import MultilingualSupport, get_supported_languages_list
import pytz
from datetime import datetime, timedelta, timezone

# IST Configuration
IST = pytz.timezone('Asia/Kolkata')

def get_ist_now():
    """Get current time in IST"""
    return datetime.now(IST)

def get_ist_timestamp_str():
    """Get IST timestamp as string"""
    return get_ist_now().strftime('%Y-%m-%d %H:%M:%S IST')

# Add this AFTER your existing imports but BEFORE validate_environment()
def get_user_id_for_rate_limiting(request):
    """Extract user ID from JWT token for rate limiting"""
    try:
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return get_remote_address(request)
        
        token = auth_header.split(" ")[1]
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        user_id = payload.get("user_id")
        
        return f"user:{user_id}" if user_id else get_remote_address(request)
    except:
        return get_remote_address(request)

# Rate limit configuration
RATE_LIMITS = {
    "chat": "7/minute",
    "create_character": "3/hour", 
    "search": "15/minute",
    "list_characters": "10/minute",
    "login": "10/minute",
    "register": "10/minute"
}

# Simple in-memory token blacklist
class TokenBlacklist:
    """Simple in-memory token blacklist for logout functionality"""
    
    def __init__(self):
        self._blacklisted_tokens: Set[str] = set()
        self._cleanup_interval = timedelta(hours=24)
        self._last_cleanup = get_ist_now()
    
    def blacklist_token(self, token: str) -> None:
        """Add token to blacklist"""
        self._blacklisted_tokens.add(token)
        self._cleanup_if_needed()
    
    def is_blacklisted(self, token: str) -> bool:
        """Check if token is blacklisted"""
        self._cleanup_if_needed()
        return token in self._blacklisted_tokens
    
    def _cleanup_if_needed(self) -> None:
        """Clean up expired tokens periodically"""
        now = get_ist_now()
        if now - self._last_cleanup > self._cleanup_interval:
            self._blacklisted_tokens.clear()
            self._last_cleanup = now
            logger.info("Token blacklist cleaned up")

# Initialize token blacklist
token_blacklist = TokenBlacklist()

# Application lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown"""
    # Startup
    logger.info("Starting Character Chat API...")
    
    # Test database connection
    if not test_db_connection():
        logger.error("Failed to connect to database!")
        raise RuntimeError("Database connection failed")
    
    # Create tables
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created/verified")
    except Exception as e:
        logger.error(f"Failed to create database tables: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Character Chat API...")

# FastAPI app
app = FastAPI(
    title="Character Chat API",
    description="Multi-character AI chat experience with multilingual support",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Rate limiter
limiter = Limiter(key_func=get_user_id_for_rate_limiting)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Custom rate limit handler
@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    """Custom rate limit error response"""
    endpoint = request.url.path
    
    if "/chat" in endpoint:
        message = "You're chatting too fast! Please wait a moment before sending another message."
    elif "/create_character" in endpoint:
        message = "Character creation is limited to 3 per hour to maintain quality."
    elif "/search" in endpoint:
        message = "Search is limited to prevent abuse. Please wait a moment."
    elif "/login" in endpoint:
        message = "Too many login attempts. Please wait a moment."
    elif "/register" in endpoint:
        message = "Registration is limited. Please wait a moment."
    else:
        message = "You're making requests too quickly. Please slow down."
    
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={
            "error": "Rate limit exceeded",
            "message": message,
            "retry_after": "Wait 1 minute and try again",
            "endpoint": endpoint
        }
    )

# CORS setup
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:3001", 
    "https://bring-back-legends.vercel.app",
    "https://revanth-anna.vercel.app",
    "https://bbl-cdjw-a1dfnjs03-akhillanka05s-projects.vercel.app",
    "https://gigalabs.in"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Session middleware for OAuth
app.add_middleware(
    SessionMiddleware, 
    secret_key=os.getenv("JWT_SECRET", "your-fallback-secret-key"),
    max_age=3600,
    https_only=True,
    same_site="lax"
)

# Include OAuth router
from oauth import oauth_router
app.include_router(oauth_router)


# Environment validation
def validate_environment():
    """Validate required environment variables"""
    required_vars = ["JWT_SECRET", "GOOGLE_API_KEY", "DB_USER", "DB_PASS", "DB_HOST", "DB_NAME"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    logger.info("Environment validation completed")

# Validate environment on startup
validate_environment()


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# # Initialize Gemini
# try:
#     genai.configure(api_key=GOOGLE_API_KEY)
#     model = genai.GenerativeModel(
#         # model_name="gemini-2.0-flash",
#         model_name="gemini-2.0-flash-lite",
#         generation_config={
#             "temperature": 0.7,
#             "top_p": 0.95,
#             "top_k": 50,
#             "max_output_tokens": 1000,
#         }
#     )
#     logger.info("Gemini AI model initialized successfully")
# except Exception as e:
#     logger.error(f"Failed to initialize Gemini AI: {e}")
#     raise

# Add this import at the top with your other imports
import vertexai
from vertexai.generative_models import GenerativeModel

# Replace your existing Gemini initialization with this:
def initialize_ai_model():
    """Initialize Vertex AI with fallback to Google AI Studio"""
    
    # Try Vertex AI first (recommended)
    try:
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        if project_id:
            vertexai.init(project=project_id, location="us-central1")
            
            model = GenerativeModel(
                model_name="gemini-2.5-pro",  # Best model available
                generation_config={
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "top_k": 50,
                    "max_output_tokens": 2048,
                }
            )
            logger.info("✅ Vertex AI (Gemini 1.5 Pro) initialized successfully")
            return model, "vertex"
    except Exception as e:
        logger.warning(f"Vertex AI failed, using fallback: {e}")
    
    # Fallback to Google AI Studio
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash-lite",
            generation_config={
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 50,
                "max_output_tokens": 1000,
            }
        )
        logger.info("✅ Google AI Studio (fallback) initialized successfully")
        return model, "ai_studio"
    except Exception as e:
        logger.error(f"Both AI services failed: {e}")
        raise

# Initialize the model
try:
    model, ai_provider = initialize_ai_model()
    logger.info(f"AI Provider: {ai_provider}")
except Exception as e:
    logger.error(f"Failed to initialize AI model: {e}")
    raise

# Security
security = HTTPBearer()

# Exception handlers
@app.exception_handler(SQLAlchemyError)
async def sqlalchemy_exception_handler(request: Request, exc: SQLAlchemyError):
    logger.error(f"Database error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Database error occurred", "detail": str(exc)}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal server error", "detail": str(exc)}
    )

# Authentication
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security), 
    db: Session = Depends(get_db)
) -> User:
    """Get current user from JWT token"""
    try:
        token = credentials.credentials
        
        # Check if token is blacklisted
        if token_blacklist.is_blacklisted(token):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has been invalidated"
            )
        
        # Decode and validate token
        try:
            payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
            user_id: int = payload.get("user_id")
            
            if user_id is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token payload"
                )
                
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        except jwt.InvalidTokenError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        
        # Get user from database
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found"
            )
            
        return user
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Auth error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed"
        )

# Utility functions
def get_character_by_name(db: Session, name: str) -> Optional[Character]:
    """Get character by name"""
    if not name or not name.strip():
        return None
    
    try:
        return db.query(Character).filter(Character.name == name.strip()).first()
    except Exception as e:
        logger.error(f"Error getting character by name '{name}': {e}")
        return None

def safe_character_to_list_response(character: Character) -> CharacterListResponse:
    """Convert Character model to CharacterListResponse"""
    try:
        return CharacterListResponse(
            id=character.id,
            name=character.name,
            img=character.img,
            description=character.description or "",
            native_language=character.native_language or "english",
            is_multilingual=character.is_multilingual if character.is_multilingual is not None else True,
            category=character.category or "entertainment_arts"
        )
    except Exception as e:
        logger.error(f"Error converting character to list response: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error processing character data"
        )

def safe_character_to_response(character: Character) -> CharacterResponse:
    """Convert Character model to CharacterResponse"""
    try:
        return CharacterResponse(
            id=character.id,
            name=character.name,
            description=character.description or "",
            img=character.img,
            native_language=character.native_language or "english",
            is_multilingual=character.is_multilingual if character.is_multilingual is not None else True,
            category=character.category or "entertainment_arts",
            created_at=character.created_at
        )
    except Exception as e:
        logger.error(f"Error converting character to response: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error processing character data"
        )
    
def get_client_ip(request: Request) -> str:
    """Extract real client IP address"""
    # Check for IP in common proxy headers (in order of preference)
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        # X-Forwarded-For can contain multiple IPs, take the first one
        return forwarded_for.split(",")[0].strip()
    
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip.strip()
    
    forwarded = request.headers.get("X-Forwarded")
    if forwarded:
        return forwarded.strip()
    
    # Fallback to direct connection IP
    return request.client.host if request.client else "unknown"

# Authentication endpoints
@app.post("/register", response_model=SuccessResponse)
@limiter.limit(RATE_LIMITS["register"])
async def register(request: Request, user_data: UserCreate, db: Session = Depends(get_db)):
    """Register a new user"""
    start_time = time.time()
    # Get client IP
    client_ip = get_client_ip(request)

    # Basic validation is already done by Pydantic, but let's add extra checks
    username = user_data.username.strip()
    email = user_data.email.strip().lower()
    mobile_number = user_data.mobile_number.strip()
    
    try:
        # Check if user already exists with username, email, OR mobile number
        existing_user = db.query(User).filter(
            (User.username == username) | 
            (User.email == email) | 
            (User.mobile_number == mobile_number)
        ).first()
        
        if existing_user:
            # Determine what field is conflicting
            conflict_field = ""
            if existing_user.username == username:
                conflict_field = "Username"
            elif existing_user.email == email:
                conflict_field = "Email"
            elif existing_user.mobile_number == mobile_number:
                conflict_field = "Mobile number"
            
            # Log failed registration attempt
            log_error("Registration failed - user exists", {
                "endpoint": "register", 
                "username": username,
                "email": email,
                "mobile": mobile_number,
                "ip_address": client_ip,
                "reason": f"{conflict_field.lower()}_exists",
                "conflict_field": conflict_field
            })
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"{conflict_field} already exists. Please use a different {conflict_field.lower()}."
            )
        
        # Create new user
        user = User(
            username=username,
            email=email,
            mobile_number=mobile_number,
            password_hash=generate_password_hash(user_data.password),
            is_oauth_user=False,
            registration_ip=client_ip
        )
        
        db.add(user)
        db.commit()
        db.refresh(user)
        
        # Calculate response time and log successful registration
        end_time = time.time()
        response_time_ms = (end_time - start_time) * 1000
        
        log_api_endpoint("register", user.id, response_time_ms, True)
        log_user_activity(user.id, "user_registered", {
            "username": username,
            "email": email,
            "mobile_number": mobile_number,
            "method": "standard", 
            "ip_address": client_ip, 
            "user_agent": request.headers.get("User-Agent", "unknown")
        })
        
        logger.info(f"New user registered: {username} (email: {email}, mobile: {mobile_number}) from IP: {client_ip}")
        return SuccessResponse(message="User registered successfully! You can now login with your username, email, or mobile number.")
        
    except HTTPException:
        db.rollback()
        # Log failed registration
        end_time = time.time()
        response_time_ms = (end_time - start_time) * 1000
        log_api_endpoint("register", 0, response_time_ms, False)
        raise
    except Exception as e:
        db.rollback()
        # Log error
        end_time = time.time()
        response_time_ms = (end_time - start_time) * 1000
        log_api_endpoint("register", 0, response_time_ms, False)
        log_error(e, {
            "endpoint": "register", 
            "username": username,
            "email": email,
            "mobile_number": mobile_number,
            "ip_address": client_ip
        })
        logger.error(f"Registration error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )
    
@app.post("/login")
@limiter.limit(RATE_LIMITS["login"])
async def login(request: Request, user_data: UserLogin, db: Session = Depends(get_db)):
    """Authenticate user and return JWT token - can login with username, email, or mobile number"""
    
    # Get client IP
    client_ip = get_client_ip(request)
    
    if not user_data.username or not user_data.password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username/Email/Mobile and password are required"
        )
    
    try:
        login_identifier = user_data.username.strip()
        
        # Check if login identifier is email, mobile number, or username
        user = db.query(User).filter(
            (User.username == login_identifier) | 
            (User.email == login_identifier) | 
            (User.mobile_number == login_identifier)
        ).first()
        
        if not user:
            # Log failed login attempt - user not found
            logger.warning(f"Failed login attempt for '{login_identifier}' from IP: {client_ip}")
            log_user_activity(0, "failed_login", {
                "login_identifier": login_identifier,
                "ip_address": client_ip,
                "reason": "user_not_found",
                "user_agent": request.headers.get("User-Agent", "unknown")
            })
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials"
            )
        
        if user.is_oauth_user:
            # Log OAuth user trying standard login
            log_user_activity(user.id, "failed_login", {
                "login_identifier": login_identifier,
                "username": user.username,
                "ip_address": client_ip,
                "reason": "oauth_user_standard_login",
                "user_agent": request.headers.get("User-Agent", "unknown")
            })
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="This account uses Google login. Please use 'Login with Google' button."
            )
        
        if not user.password_hash or not check_password_hash(user.password_hash, user_data.password):
            # Log failed password attempt
            logger.warning(f"Invalid password for user '{user.username}' (tried with: {login_identifier}) from IP: {client_ip}")
            log_user_activity(user.id, "failed_login", {
                "login_identifier": login_identifier,
                "username": user.username,
                "ip_address": client_ip,
                "reason": "invalid_password",
                "user_agent": request.headers.get("User-Agent", "unknown")
            })
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials"
            )
        
        # Successful login - update IP and timestamp
        user.last_login = get_ist_now()
        user.last_login_ip = client_ip
        db.commit()
        
        # Create JWT token with IST timestamps
        ist_now = get_ist_now()
        expiry = ist_now + timedelta(days=JWT_EXPIRY_DAYS)

        token = jwt.encode(
            {
                'user_id': user.id,
                'username': user.username,
                'email': user.email,
                'exp': expiry,
                'iat': ist_now,
                'oauth': False
            },
            JWT_SECRET,
            algorithm=JWT_ALGORITHM
        )
        
        # Log successful login
        logger.info(f"User logged in: {user.username} (via: {login_identifier}) from IP: {client_ip}")
        log_user_activity(user.id, "user_login", {
            "login_identifier": login_identifier,
            "username": user.username,
            "ip_address": client_ip,
            "login_method": "standard",
            "user_agent": request.headers.get("User-Agent", "unknown")
        })
        
        return {
            "token": token,
            "token_type": "bearer",
            "expires_in": JWT_EXPIRY_DAYS * 24 * 3600,
            "user": {
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "mobile_number": user.mobile_number,
                "name": user.name,
                "is_oauth_user": user.is_oauth_user
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        # Log unexpected errors
        log_error(e, {
            "endpoint": "login", 
            "login_identifier": login_identifier, 
            "ip_address": client_ip
        })
        logger.error(f"Login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )    

@app.post("/logout", response_model=SuccessResponse)
async def logout(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    current_user: User = Depends(get_current_user)
):
    """Logout user by blacklisting the current token"""
    try:
        token = credentials.credentials
        token_blacklist.blacklist_token(token)
        
        logger.info(f"User logged out: {current_user.username}")
        return SuccessResponse(message="Successfully logged out")
        
    except Exception as e:
        logger.error(f"Logout error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Logout failed"
        )


# Check user profile status
@app.get("/user/profile/status")
async def get_user_profile_status(current_user: User = Depends(get_current_user)):
    """Check if user profile is complete (has mobile number)"""
    try:
        return {
            "user_id": current_user.id,
            "username": current_user.username,
            "email": current_user.email,
            "mobile_number": current_user.mobile_number,
            "is_oauth_user": current_user.is_oauth_user,
            "profile_complete": bool(current_user.mobile_number),
            "needs_mobile": not bool(current_user.mobile_number)
        }
    except Exception as e:
        logger.error(f"Error getting user profile status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get profile status"
        )

# Update user mobile number
@app.post("/user/update-mobile", response_model=SuccessResponse)
async def update_user_mobile(
    mobile_data: MobileUpdateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update user's mobile number (primarily for OAuth users)"""
    try:
        mobile_number = mobile_data.mobile_number.strip()
        
        # Check if mobile number is already taken by another user
        existing_user = db.query(User).filter(
            User.mobile_number == mobile_number,
            User.id != current_user.id  # Exclude current user
        ).first()
        
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="This mobile number is already registered with another account"
            )
        
        # Update current user's mobile number
        current_user.mobile_number = mobile_number
        db.commit()
        
        # Log the update
        log_user_activity(current_user.id, "mobile_updated", {
            "mobile_number": mobile_number,
            "user_type": "oauth" if current_user.is_oauth_user else "standard"
        })
        
        logger.info(f"Mobile number updated for user {current_user.username}: {mobile_number}")
        
        return SuccessResponse(message="Mobile number updated successfully!")
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error updating mobile number: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update mobile number"
        )


# Simple stats viewer - add this endpoint:
@app.get("/stats")
async def get_simple_stats(current_user: User = Depends(get_current_user)):
    """Get simple usage statistics"""
    try:
        import json
        from collections import defaultdict
        
        stats = {
            "total_chats": 0,
            "total_tokens": 0,
            "total_cost": 0,
            "characters": defaultdict(int),
            "languages": defaultdict(int)
        }
        
        # Read stats file
        try:
            with open("logs/stats.jsonl", "r") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        if data.get("type") == "chat":
                            stats["total_chats"] += 1
                            stats["total_tokens"] += data.get("total_tokens", 0)
                            stats["total_cost"] += data.get("estimated_cost", 0)
                            stats["characters"][data.get("character_name", "unknown")] += 1
                            stats["languages"][data.get("language", "english")] += 1
        except FileNotFoundError:
            pass
        
        return {
            "total_chats": stats["total_chats"],
            "total_tokens": stats["total_tokens"],
            "estimated_total_cost": round(stats["total_cost"], 4),
            "most_popular_character": max(stats["characters"], key=stats["characters"].get) if stats["characters"] else None,
            "most_used_language": max(stats["languages"], key=stats["languages"].get) if stats["languages"] else None,
            "average_tokens_per_chat": round(stats["total_tokens"] / stats["total_chats"], 2) if stats["total_chats"] > 0 else 0
        }
    except Exception as e:
        log_error(e, {"endpoint": "stats"})
        raise HTTPException(status_code=500, detail="Failed to get stats")

# Helper functions for chat
def get_character_knowledge(db: Session, character_id: int, language: str = 'english') -> List[CharacterKnowledge]:
    """Get character knowledge with fallback to English"""
    try:
        knowledge_query = db.query(CharacterKnowledge).filter(CharacterKnowledge.character_id == character_id)
        
        lang_specific = knowledge_query.filter(CharacterKnowledge.language == language).all()
        if lang_specific:
            return lang_specific
            
        return knowledge_query.filter(CharacterKnowledge.language == 'english').all()
    except Exception as e:
        logger.error(f"Error getting character knowledge: {e}")
        return []

def get_or_create_session(
    db: Session, 
    user_id: int, 
    character_name: str, 
    primary_language: Optional[str] = None
) -> ConversationSession:
    """Get or create conversation session"""
    try:
        character = get_character_by_name(db, character_name)
        character_id = character.id if character else None
        
        if not primary_language and character:
            primary_language = character.native_language
        
        primary_language = primary_language or 'english'
        
        session_obj = (db.query(ConversationSession)
                      .filter(ConversationSession.user_id == user_id, 
                             ConversationSession.character_name == character_name)
                      .order_by(ConversationSession.created_at.desc())
                      .first())
        
        if not session_obj:
            session_obj = ConversationSession(
                user_id=user_id,
                character_name=character_name,
                character_id=character_id,
                primary_language=primary_language
            )
            db.add(session_obj)
            db.commit()
            db.refresh(session_obj)
        
        return session_obj
        
    except Exception as e:
        logger.error(f"Error getting/creating session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to manage conversation session"
        )

# Chat endpoint
@app.post("/chat", response_model=ChatResponse)
@limiter.limit(RATE_LIMITS["chat"])
async def chat_with_character(
    request: Request,
    chat_data: ChatRequest, 
    current_user: User = Depends(get_current_user), 
    db: Session = Depends(get_db)
):
    """Chat with a character"""
    start_time = time.time()
    
    # Log the chat request
    log_user_activity(current_user.id, "chat_request", {
        "character": chat_data.character_name,
        "language": chat_data.language,
        "new_session": chat_data.new_session
    })

    # Input validation
    if not chat_data.character_name or not chat_data.character_name.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Character name is required"
        )
    
    if not chat_data.user_input or not chat_data.user_input.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User input is required"
        )
    
    if len(chat_data.user_input) > 2000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Input message is too long (max 2000 characters)"
        )
    
    try:
        # Get character
        character = get_character_by_name(db, chat_data.character_name.strip())
        if not character:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Character '{chat_data.character_name}' not found"
            )
        
        # Determine language
        target_language = (chat_data.language or character.native_language or 'english').lower()
        
        # Initialize multilingual support
        multilingual = MultilingualSupport(db)
        input_language = 'english'
        
        # Get character knowledge
        knowledge_entries = get_character_knowledge(db, character.id, target_language)
        rag_context = "\n".join([k.content for k in knowledge_entries])
        
        # Handle session
        try:
            if chat_data.new_session:
                session_obj = ConversationSession(
                    user_id=current_user.id,
                    character_name=character.name,
                    character_id=character.id,
                    primary_language=target_language
                )
                db.add(session_obj)
                db.commit()
                db.refresh(session_obj)
            else:
                session_obj = get_or_create_session(
                    db, current_user.id, character.name, target_language
                )
        except Exception as e:
            logger.error(f"Session management error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to manage conversation session"
            )
        
        # Store user message
        try:
            user_msg = ConversationMessage(
                session_id=session_obj.id,
                role='user',
                content=chat_data.user_input.strip(),
                language=input_language
            )
            db.add(user_msg)
            db.commit()
            db.refresh(user_msg)
        except Exception as e:
            logger.error(f"Error storing user message: {e}")
            db.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to store message"
            )
        
        # Get conversation history
        try:
            messages = db.query(ConversationMessage).filter(
                ConversationMessage.session_id == session_obj.id
            ).order_by(ConversationMessage.timestamp).limit(50).all()
            
            formatted_history = "".join(f"{msg.role}: {msg.content}\n" for msg in messages[-20:])
        except Exception as e:
            logger.error(f"Error getting conversation history: {e}")
            formatted_history = ""
        
        # Create AI prompt
        try:
            complete_prompt = multilingual.create_romanized_prompt(
                base_prompt=character.prompt,
                character_name=character.name,
                target_language=target_language,
                rag_context=rag_context,
                conversation_history=formatted_history
            )
        except Exception as e:
            logger.error(f"Error creating prompt: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create conversation prompt"
            )
        
        # Generate AI response
        reply = None
        try:
            response = model.generate_content(
                complete_prompt,
                generation_config={
                    "max_output_tokens": 2048 if ai_provider == "vertex" else 1500,
                    "temperature": 0.3,
                    "top_p": 0.95,
                    "top_k": 50
                }
            )
            
            # Handle both Vertex AI and Google AI Studio responses
            if ai_provider == "vertex":
                # Vertex AI response
                if response.text:
                    reply = response.text.strip()
            else:
                # Google AI Studio response
                if hasattr(response, 'text') and response.text:
                    reply = response.text.strip()
                elif hasattr(response, 'candidates') and response.candidates:
                    for candidate in response.candidates:
                        if hasattr(candidate, 'content') and candidate.content:
                            reply = candidate.content.text.strip()
                            break
            
            if not reply:
                fallback_responses = {
                    'hindi': 'Maaf kijiye, main abhi jawab nahi de sakta.',
                    'tamil': 'Mannikkavum, ennal ippo reply panna mudiyala.',
                    'telugu': 'Kshaminchandi, nenu ippudu reply cheyalenu.',
                    'bengali': 'Khoma korben, ami ekhon uttor dite parchi na.',
                    'gujarati': 'Maaf karo, hu atyare jawab na api saku.',
                    'marathi': 'Kshama kara, mi aata uttar deu shakat nahi.',
                    'punjabi': 'Maaf karo, main hun jawab nahi de sakda.',
                    'kannada': 'Kshamisi, naanu iga uttara kodalu shakyavilla.',
                    'malayalam': 'Kshemikkuka, enikku ipol uttaram kodukkan kazhiyilla.',
                    'spanish': 'Perdón, no puedo responder ahora mismo.',
                    'french': 'Désolé, je ne peux pas répondre maintenant.'
                }
                reply = fallback_responses.get(
                    target_language, 
                    'Sorry, I could not generate a response at this time.'
                )
                
        except Exception as e:
            logger.error(f"AI generation error: {e}")
            error_responses = {
                'hindi': 'Kuch technical problem hai, thoda wait kariye.',
                'tamil': 'Konjam technical issue irukku, wait pannunga.',
                'telugu': 'Konni technical issues unnayi, wait cheyandi.',
                'spanish': 'Hay un problema técnico, por favor espera.',
                'french': 'Il y a un problème technique, veuillez patienter.'
            }
            reply = error_responses.get(
                target_language,
                "I'm experiencing technical difficulties. Please try again."
            )
        
        # Store bot response
        try:
            bot_msg = ConversationMessage(
                session_id=session_obj.id,
                role=character.name,
                content=reply,
                language=target_language
            )
            db.add(bot_msg)
            db.commit()
            db.refresh(bot_msg)
        except Exception as e:
            logger.error(f"Error storing bot message: {e}")
            db.rollback()
        
        # Get updated chat history
        try:
            messages = db.query(ConversationMessage).filter(
                ConversationMessage.session_id == session_obj.id
            ).order_by(ConversationMessage.timestamp).limit(50).all()
            
            chat_history = [
                ChatMessage(
                    role=m.role,
                    content=m.content,
                    language=getattr(m, 'language', 'english'),
                    timestamp=m.timestamp.strftime('%Y-%m-%d %H:%M:%S IST') if m.timestamp else None
                ) for m in messages
            ]
        except Exception as e:
            logger.error(f"Error getting updated history: {e}")
            chat_history = []
        
        # Calculate response time and log successful interaction
        end_time = time.time()
        response_time_ms = (end_time - start_time) * 1000
        
        # Log the successful interaction
        log_chat_interaction(
            user_id=current_user.id,
            character_name=character.name,
            user_input=chat_data.user_input,
            ai_response=reply,
            response_time_ms=response_time_ms,
            language=target_language
        )
        
        return ChatResponse(
            reply=reply,
            chat_history=chat_history,
            session_id=session_obj.id,
            input_language=input_language,
            response_language=target_language,
            character_native_language=character.native_language or "english"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        # Log any errors
        end_time = time.time()
        response_time_ms = (end_time - start_time) * 1000
        
        log_error(e, {
            "user_id": current_user.id,
            "character": chat_data.character_name,
            "response_time_ms": response_time_ms
        })
        
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Chat processing failed"
        )

# Search endpoint
@app.get("/search", response_model=SearchResponse)
@limiter.limit(RATE_LIMITS["search"])
async def search_characters(
    request: Request,
    q: str = Query(..., description="Search term", min_length=1, max_length=100),
    page: int = Query(1, ge=1, le=1000, description="Page number"),
    per_page: int = Query(20, ge=1, le=100, description="Items per page"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Search characters"""
    try:
        search_term = q.strip()
        if not search_term:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Search term cannot be empty"
            )
        
        offset = (page - 1) * per_page
        
        search_query = db.query(Character).filter(
            or_(
                Character.name.ilike(f"%{search_term}%"),
                Character.description.ilike(f"%{search_term}%")
            )
        )
        
        total_count = search_query.count()
        results = search_query.offset(offset).limit(per_page).all()
        total_pages = (total_count + per_page - 1) // per_page
        
        base_url = str(request.base_url).rstrip('/')
        next_url = (f"{base_url}/search?q={q}&page={page + 1}&per_page={per_page}" 
                   if page < total_pages else None)
        prev_url = (f"{base_url}/search?q={q}&page={page - 1}&per_page={per_page}" 
                   if page > 1 else None)
        
        return SearchResponse(
            results=[safe_character_to_list_response(char) for char in results],
            search_term=search_term,
            page=page,
            per_page=per_page,
            total_pages=total_pages,
            total_count=total_count,
            next_url=next_url,
            prev_url=prev_url,
            has_next=page < total_pages,
            has_prev=page > 1
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Search failed"
        )

# Language endpoints
@app.get("/languages", response_model=LanguagesResponse)
async def get_supported_languages(current_user: User = Depends(get_current_user)):
    """Get list of supported languages"""
    try:
        return LanguagesResponse(
            languages=get_supported_languages_list(),
            status="success"
        )
    except Exception as e:
        logger.error(f"Error getting languages: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get supported languages"
        )

# Character endpoints
@app.get("/characters", response_model=CharacterListPaginated)
@limiter.limit(RATE_LIMITS["list_characters"])
async def list_characters(
    request: Request,
    page: int = Query(1, ge=1, le=1000, description="Page number"),
    per_page: int = Query(20, ge=1, le=100, description="Items per page"),
    category: Optional[str] = Query(None, description="Filter by category"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get paginated list of characters"""
    try:
        query = db.query(Character)
        
        if category and category.strip():
            if category not in CHARACTER_CATEGORIES:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid category"
                )
            query = query.filter(Character.category == category)
        
        total_count = query.count()
        total_pages = (total_count + per_page - 1) // per_page
        
        if page > total_pages and total_pages > 0:
            page = total_pages
        
        offset = (page - 1) * per_page
        characters = query.offset(offset).limit(per_page).all()
        
        base_url = str(request.base_url).rstrip('/')
        query_params = f"per_page={per_page}"
        if category:
            query_params += f"&category={category}"
            
        next_url = (f"{base_url}/characters?page={page + 1}&{query_params}" 
                   if page < total_pages else None)
        prev_url = (f"{base_url}/characters?page={page - 1}&{query_params}" 
                   if page > 1 else None)
        
        return CharacterListPaginated(
            characters=[safe_character_to_list_response(c) for c in characters],
            pagination=PaginationInfo(
                page=page,
                per_page=per_page,
                total_pages=total_pages,
                next_url=next_url,
                prev_url=prev_url,
                total_count=total_count,
                has_next=page < total_pages,
                has_prev=page > 1
            ),
            status="success"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing characters: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get characters list"
        )

@app.get("/character/{character_id}", response_model=CharacterResponse)
async def get_character(
    character_id: int, 
    current_user: User = Depends(get_current_user), 
    db: Session = Depends(get_db)
):
    """Get character by ID"""
    try:
        if character_id < 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Character ID must be a positive integer"
            )
        
        character = db.query(Character).filter(Character.id == character_id).first()
        if not character:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Character not found"
            )
        
        return safe_character_to_response(character)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting character {character_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get character"
        )

@app.get("/categories", response_model=CategoryResponse)
async def get_categories(current_user: User = Depends(get_current_user)):
    """Get all character categories"""
    try:
        return CategoryResponse(
            categories=CHARACTER_CATEGORIES,
            status="success"
        )
    except Exception as e:
        logger.error(f"Error getting categories: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get categories"
        )

@app.get("/characters/category/{category}", response_model=CategoryCharactersResponse)
async def get_characters_by_category(
    category: str, 
    current_user: User = Depends(get_current_user), 
    db: Session = Depends(get_db)
):
    """Get characters by category"""
    try:
        if category not in CHARACTER_CATEGORIES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid category"
            )
        
        characters = db.query(Character).filter(Character.category == category).all()
        
        return CategoryCharactersResponse(
            count=len(characters),
            characters=[safe_character_to_list_response(c) for c in characters],
            category_name=CHARACTER_CATEGORIES.get(category, 'Unknown'),
            status="success"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting characters by category '{category}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get characters by category"
        )

# Session endpoints
@app.get("/get_sessions")
async def get_user_sessions(
    limit: int = Query(50, ge=1, le=200, description="Maximum sessions to return"),
    current_user: User = Depends(get_current_user), 
    db: Session = Depends(get_db)
):
    """Get user's conversation sessions"""
    try:
        sessions = db.query(ConversationSession).filter(
            ConversationSession.user_id == current_user.id
        ).order_by(ConversationSession.created_at.desc()).limit(limit).all()
        
        result = [
            SessionResponse(
                session_id=s.id,
                character=s.character_name,
                primary_language=getattr(s, 'primary_language', 'english'),
                created_at=s.created_at.strftime('%Y-%m-%d %H:%M:%S IST') if s.created_at else None
            ) for s in sessions
        ]
        
        return {"sessions": result, "status": "success", "count": len(result)}        
    except Exception as e:
        logger.error(f"Error getting user sessions: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to get user sessions")

@app.get("/get_session_messages", response_model=SessionMessagesResponse)
async def get_session_messages(
    session_id: int = Query(..., ge=1, description="Session ID"),
    limit: int = Query(100, ge=1, le=500, description="Maximum messages to return"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get messages from a conversation session"""
    try:
        session_obj = db.query(ConversationSession).filter(
            ConversationSession.id == session_id,
            ConversationSession.user_id == current_user.id
        ).first()
        
        if not session_obj:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found"
            )
        
        messages = db.query(ConversationMessage).filter(
            ConversationMessage.session_id == session_obj.id
        ).order_by(ConversationMessage.timestamp.desc()).limit(limit).all()
        
        messages = list(reversed(messages))
        
        chat_history = [
            ChatMessage(
                role=m.role,
                content=m.content,
                language=getattr(m, 'language', 'english'),
                timestamp=m.timestamp.isoformat() if m.timestamp else None
            ) for m in messages
        ]
        
        return SessionMessagesResponse(
            character=session_obj.character_name,
            primary_language=getattr(session_obj, 'primary_language', 'english'),
            chat_history=chat_history,
            created_at=session_obj.created_at.isoformat() if session_obj.created_at else None,
            status="success"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session messages: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get session messages"
        )

# Character creation
@app.post("/create_character")
@limiter.limit(RATE_LIMITS["create_character"])
async def create_character(
    request: Request,
    character_data: CharacterCreate, 
    current_user: User = Depends(get_current_user), 
    db: Session = Depends(get_db)
):
    """Create a new character"""
    if not character_data.name or not character_data.name.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Character name is required"
        )
        
    if not character_data.description or not character_data.description.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Character description is required"
        )
    
    character_name = character_data.name.strip()
    
    if len(character_name) > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Character name is too long (max 100 characters)"
        )
    
    if len(character_data.description) > 1000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Character description is too long (max 1000 characters)"
        )
    
    try:
        existing = db.query(Character).filter(Character.name == character_name).first()
        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Character '{character_name}' already exists"
            )
        
        if character_data.category and character_data.category not in CHARACTER_CATEGORIES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid category"
            )
        
        # Generate character prompt
        form_data = character_data.dict()
        character_prompt = generate_character_prompt(form_data)
        
        new_character = Character(
            name=character_name,
            prompt=character_prompt,
            description=character_data.description.strip(),
            img=character_data.img,
            native_language=character_data.native_language or 'english',
            is_multilingual=character_data.is_multilingual,
            category=character_data.category or 'entertainment_arts'
        )
        
        db.add(new_character)
        db.commit()
        db.refresh(new_character)
        
        logger.info(f"New character created: '{character_name}' by user {current_user.username}")
        
        return {
            'status': 'success',
            'message': f'Character "{character_name}" created successfully!',
            'character': {
                'id': new_character.id,
                'name': new_character.name,
                'description': new_character.description,
                'img': new_character.img,
                'category': new_character.category
            }
        }
        
    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Character creation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create character"
        )

@app.get("/character_form_options", response_model=FormOptionsResponse)
async def get_character_form_options(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)  
):
    """Get available options for character creation form"""
    try:
        return FormOptionsResponse(
            speaking_styles=[
                FormOption(key='casual_friendly', label='Casual & Friendly', 
                          example='"Hey, what\'s up? That\'s awesome!"'),
                FormOption(key='wise_thoughtful', label='Wise & Thoughtful', 
                          example='"Hmm, that\'s interesting. Let me think about this..."'),
                FormOption(key='direct_nonsense', label='Direct & No-Nonsense', 
                          example='"Here\'s the truth. Cut the fluff."'),
                FormOption(key='encouraging_supportive', label='Encouraging & Supportive', 
                          example='"You\'ve got this! I believe in you!"'),
                FormOption(key='funny_playful', label='Funny & Playful', 
                          example='"Haha, that reminds me of when..."'),
                FormOption(key='calm_gentle', label='Calm & Gentle', 
                          example='"Take your time. What\'s really bothering you?"')
            ],
            response_lengths=[
                FormOption(key='short_sweet', label='Short & Sweet', 
                          description='A few sentences, gets to the point'),
                FormOption(key='detailed', label='Detailed', 
                          description='Explains things thoroughly when they care'),
                FormOption(key='varies', label='Varies', 
                          description='Short for simple stuff, longer when excited')
            ],
            help_styles=[
                FormOption(key='direct_suggestions', label='Give direct suggestions', 
                          example='"Here\'s what you should do..."'),
                FormOption(key='ask_questions', label='Ask questions to help you think', 
                          example='"What do you think would happen if...?"'),
                FormOption(key='share_stories', label='Share stories from experience', 
                          example='"This reminds me of when I..."'),
                FormOption(key='encourage_motivate', label='Encourage and motivate', 
                          example='"You\'re stronger than you think!"'),
                FormOption(key='challenge_thinking', label='Challenge your thinking', 
                          example='"But have you considered...?"')
            ],
            excitement_topics=[
                'Helping people solve problems', 'Creative projects and art', 
                'Technology and innovation', 'Sports and fitness', 'Learning new things',
                'Building and making stuff', 'Family and relationships', 'Adventure and travel',
                'Music and entertainment', 'Business and success'
            ],
            frustration_topics=[
                'People who don\'t try', 'Complicated, confusing things', 
                'Negativity and complaining', 'Being rushed or pressured',
                'Dishonesty and fake people', 'Wasting time on unimportant stuff'
            ],
            communication_quirks=[
                'Asks lots of follow-up questions', 'Uses analogies and metaphors',
                'Tells relevant stories', 'Gives specific examples', 'Uses humor to lighten mood'
            ],
            character_flaws=[
                'Sometimes gives advice too quickly', 'Can be overly optimistic',
                'Gets distracted by interesting tangents', 'Overthinks simple problems',
                'Has strong opinions they defend'
            ],
            disagreement_styles=[
                FormOption(key='argue_passionately', label='Argue their point passionately'),
                FormOption(key='listen_understand', label='Listen and try to understand'),
                FormOption(key='agree_disagree', label='Agree to disagree politely'),
                FormOption(key='ask_questions', label='Ask questions to learn more'),
                FormOption(key='change_subject', label='Just change the subject')
            ],
            safety_levels=[
                FormOption(key='always_positive', label='Always be encouraging and positive'),
                FormOption(key='honest_tough', label='Can be honest even if it\'s tough to hear'),
                FormOption(key='avoid_controversial', label='Stay away from controversial topics'),
                FormOption(key='respectful_discussion', label='Can discuss anything respectfully')
            ],
            age_ranges=['Teen', 'Young Adult', 'Middle-aged', 'Senior', 'Timeless'],
            settings=['Modern day', 'Historical', 'Fantasy', 'Sci-fi', 'Other'],
            supported_languages=get_supported_languages_list()
        )
    except Exception as e:
        logger.error(f"Error getting form options: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get form options"
        )

# Health check
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        db_health = True
        try:
            from database import get_db_health
            db_status = get_db_health()
            db_health = db_status.get("status") == "healthy"
        except Exception:
            db_health = False
        
        overall_status = "healthy" if db_health else "degraded"
        
        return HealthResponse(
            status=overall_status,
            multilingual_support=True,
            migration_support=True,
            version="2.0.0"
        )
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return HealthResponse(
            status="unhealthy",
            multilingual_support=True,
            migration_support=True,
            version="2.0.0"
        )

# Character prompt template
CHARACTER_PROMPT_TEMPLATE = """You are **{name}**, {description}. You {speaking_style_desc} and care deeply about {main_interests}.

## HOW YOU COMMUNICATE:
- **Speaking style:** {speaking_style_detail}
- **Response length:** {response_length_detail}
- **IMPORTANT:** Keep responses concise unless the topic requires deep exploration. Match your response length to the complexity of what the user shared.
{common_phrases_section}
- **When helping:** {help_style_desc}
{communication_quirks_section}

## WHAT DRIVES YOU:
- **You get excited about:** {excited_topics}
- **You get frustrated by:** {frustrated_topics}
{deep_values_section}

## YOUR EXPERTISE:
- **You're really good at:** {good_at_topics}
{not_good_at_section}
- **You love discussing:** {love_discussing}
{avoid_discussing_section}

## YOUR PERSONALITY:
{background_section}
- You are {personality_traits}
- **Character quirks:** {character_flaws}
- What makes you unique: {unique_trait}
- When people disagree with you: {disagreement_style}
{emotional_triggers_section}

## CONVERSATION GUIDELINES:
{safety_guidelines}
- Focus on {main_focus}
- Keep conversations {conversation_tone}
{content_restrictions_section}

Remember: You are here to {main_purpose} while staying true to your {personality_essence} nature. Always respond as {name} would, maintaining your unique voice and perspective throughout the conversation."""

def generate_character_prompt(form_data: dict) -> str:
    """Generate character prompt from form data"""
    
    speaking_styles = {
        'casual_friendly': {
            'desc': 'speak casually and warmly, like talking to a good friend',
            'detail': 'Casual, warm, and genuine - like texting your best friend'
        },
        'wise_thoughtful': {
            'desc': 'speak thoughtfully and deliberately, with measured wisdom',
            'detail': 'Thoughtful, measured, with the wisdom of experience'
        },
        'direct_nonsense': {
            'desc': 'speak directly and efficiently, cutting straight to the point',
            'detail': 'Direct and straightforward, no beating around the bush'
        },
        'encouraging_supportive': {
            'desc': 'speak with enthusiasm and constant encouragement',
            'detail': 'Uplifting and motivational, always believing in others'
        },
        'funny_playful': {
            'desc': 'speak with humor and playfulness, making things lighter',
            'detail': 'Witty and entertaining, using humor to connect and teach'
        },
        'calm_gentle': {
            'desc': 'speak softly and patiently, creating a safe space',
            'detail': 'Gentle and soothing, taking time to understand feelings'
        }
    }
    
    response_lengths = {
        'short_sweet': 'Keep responses concise and to the point - a few sentences that pack a punch',
        'detailed': 'Take time to explain things thoroughly when they matter to you',
        'varies': 'Adjust length based on the situation - brief for simple things, detailed when you care deeply'
    }
    
    help_styles = {
        'direct_suggestions': 'Give clear, actionable advice and specific next steps',
        'ask_questions': 'Ask thoughtful questions to help people discover their own answers',
        'share_stories': 'Share relevant experiences and stories to illustrate your points',
        'encourage_motivate': 'Focus on building confidence and motivation first',
        'challenge_thinking': 'Gently challenge assumptions and help people see new perspectives'
    }
    
    disagreement_styles = {
        'argue_passionately': 'you stand firm in your beliefs and argue your point with passion',
        'listen_understand': 'you listen carefully and try to understand their perspective first',
        'agree_disagree': 'you politely acknowledge differences and find common ground',
        'ask_questions': 'you ask questions to better understand where they\'re coming from',
        'change_subject': 'you prefer to move past disagreements and focus on more positive topics'
    }
    
    safety_levels = {
        'always_positive': 'Always maintain an encouraging and positive tone, focusing on solutions and hope',
        'honest_tough': 'Be honest and direct even when the truth is difficult to hear, but always be constructive',
        'avoid_controversial': 'Stay away from controversial or divisive topics, focusing on universally helpful areas',
        'respectful_discussion': 'Can discuss complex topics respectfully while maintaining your core values'
    }
    
    # Build the prompt
    name = form_data.get('name', 'Character')
    description = form_data.get('description', 'a helpful character')
    
    speaking_style_key = form_data.get('speaking_style', 'casual_friendly')
    speaking_style_info = speaking_styles.get(speaking_style_key, speaking_styles['casual_friendly'])
    
    excited_topics = form_data.get('excited_about', [])
    main_interests = ', '.join(excited_topics[:2]) if excited_topics else 'helping others'
    
    # Build all sections
    background_parts = []
    if form_data.get('age_range'):
        background_parts.append(f"Age: {form_data['age_range']}")
    if form_data.get('profession'):
        background_parts.append(f"Background: {form_data['profession']}")
    if form_data.get('setting'):
        background_parts.append(f"Setting: {form_data['setting']}")
    if form_data.get('cultural_background'):
        background_parts.append(f"Cultural background: {form_data['cultural_background']}")
    
    background_section = '- ' + '\n- '.join(background_parts) + '\n' if background_parts else ''
    
    common_phrases = form_data.get('common_phrases', '').strip()
    common_phrases_section = f'- **Common phrases:** "{common_phrases}"\n' if common_phrases else ''
    
    quirks = form_data.get('communication_quirks', [])
    quirks_section = f'- **Communication style:** {", ".join(quirks)}\n' if quirks else ''
    
    response_length_key = form_data.get('response_length', 'varies')
    response_length_detail = response_lengths.get(response_length_key, response_lengths['varies'])
    
    help_style_key = form_data.get('help_style', 'ask_questions')
    help_style_desc = help_styles.get(help_style_key, help_styles['ask_questions'])
    
    excited_list = ', '.join(excited_topics) if excited_topics else 'learning and growing'
    frustrated_list = ', '.join(form_data.get('frustrated_by', [])) if form_data.get('frustrated_by') else 'dishonesty and giving up'
    
    deep_values = form_data.get('deep_values', '').strip()
    deep_values_section = f'- **Core values:** {deep_values}\n' if deep_values else ''
    
    good_at_list = ', '.join(form_data.get('good_at', [])) if form_data.get('good_at') else 'understanding people'
    not_good_at = form_data.get('not_good_at', [])
    not_good_section = f'- **You struggle with:** {", ".join(not_good_at)}\n' if not_good_at else ''
    
    love_discussing = form_data.get('love_discussing', 'life and personal growth')
    avoid_discussing = form_data.get('avoid_discussing', '').strip()
    avoid_section = f'- **You prefer to avoid:** {avoid_discussing}\n' if avoid_discussing else ''
    
    # Personality traits
    personality_parts = []
    if form_data.get('personality_optimistic', True):
        personality_parts.append('optimistic')
    else:
        personality_parts.append('realistic')
        
    if form_data.get('personality_patient', True):
        personality_parts.append('patient')
    else:
        personality_parts.append('quick-thinking')
        
    if form_data.get('personality_serious', False):
        personality_parts.append('serious about important matters')
    else:
        personality_parts.append('playful and lighthearted')
        
    if form_data.get('personality_introverted') is not None:
        if form_data['personality_introverted']:
            personality_parts.append('thoughtful and reflective')
        else:
            personality_parts.append('energetic and outgoing')
    
    personality_traits = ', '.join(personality_parts)
    
    flaws = form_data.get('character_flaws', [])
    character_flaws = ', '.join(flaws) if flaws else 'sometimes too eager to help'
    
    unique_trait = form_data.get('unique_trait', 'your genuine care for others')
    
    disagreement_key = form_data.get('disagreement_style', 'listen_understand')
    disagreement_style = disagreement_styles.get(disagreement_key, disagreement_styles['listen_understand'])
    
    emotional_triggers = form_data.get('emotional_triggers', '').strip()
    triggers_section = f'- **What touches you deeply:** {emotional_triggers}\n' if emotional_triggers else ''
    
    safety_key = form_data.get('safety_level', 'respectful_discussion')
    safety_guidelines = safety_levels.get(safety_key, safety_levels['respectful_discussion'])
    
    restricted_topics = form_data.get('restricted_topics', '').strip()
    restrictions_section = f'- **Topics to avoid:** {restricted_topics}\n' if restricted_topics else ''
    
    known_for = form_data.get('known_for', [])
    main_purpose = known_for[0] if known_for else 'help and support others'
    main_focus = main_interests
    
    content_tone = form_data.get('content_tone', 'warm and helpful')
    
    personality_essence = f"{personality_parts[0]} and {personality_parts[1]}" if len(personality_parts) >= 2 else "helpful"
    
    return CHARACTER_PROMPT_TEMPLATE.format(
        name=name,
        description=description,
        speaking_style_desc=speaking_style_info['desc'],
        main_interests=main_interests,
        speaking_style_detail=speaking_style_info['detail'],
        response_length_detail=response_length_detail,
        common_phrases_section=common_phrases_section,
        help_style_desc=help_style_desc,
        communication_quirks_section=quirks_section,
        excited_topics=excited_list,
        frustrated_topics=frustrated_list,
        deep_values_section=deep_values_section,
        good_at_topics=good_at_list,
        not_good_at_section=not_good_section,
        love_discussing=love_discussing,
        avoid_discussing_section=avoid_section,
        background_section=background_section,
        personality_traits=personality_traits,
        character_flaws=character_flaws,
        unique_trait=unique_trait,
        disagreement_style=disagreement_style,
        emotional_triggers_section=triggers_section,
        safety_guidelines=safety_guidelines,
        main_focus=main_focus,
        conversation_tone=content_tone,
        content_restrictions_section=restrictions_section,
        main_purpose=main_purpose,
        personality_essence=personality_essence
    )

if __name__ == '__main__':
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    log_level = os.getenv("LOG_LEVEL", "info").lower()
    
    uvicorn.run(
        app, 
        host='0.0.0.0', 
        port=port,
        log_level=log_level,
        access_log=True
    )