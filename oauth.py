from fastapi import APIRouter, Request, HTTPException, Depends
from fastapi.responses import RedirectResponse
from authlib.integrations.starlette_client import OAuth
from starlette.config import Config
from sqlalchemy.orm import Session
import jwt
from datetime import datetime, timedelta, timezone
import os
import logging
from urllib.parse import urlencode
from typing import Optional
from database import get_db
from models import User
import pytz

IST = pytz.timezone('Asia/Kolkata')
def get_ist_now():
    return datetime.now(IST)

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables
config = Config('.env')

# OAuth Router
oauth_router = APIRouter(prefix="/auth", tags=["OAuth"])

# OAuth configuration
oauth = OAuth(config)

def init_oauth():
    """Initialize OAuth with Google configuration"""
    
    # Validate required environment variables
    required_vars = [
        'GOOGLE_OAUTH_CLIENT_ID',
        'GOOGLE_OAUTH_CLIENT_SECRET', 
        'GOOGLE_REDIRECT_URI',
        'FRONTEND_URL',
        'JWT_SECRET'
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        logger.error(f"Missing OAuth environment variables: {', '.join(missing_vars)}")
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    # Register Google OAuth
    oauth.register(
        name='google',
        client_id=os.getenv('GOOGLE_OAUTH_CLIENT_ID'),
        client_secret=os.getenv('GOOGLE_OAUTH_CLIENT_SECRET'),
        server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
        client_kwargs={
            'scope': 'openid email profile'
        }
    )
    
    logger.info("Google OAuth initialized successfully")
    return oauth

# Initialize OAuth on import
try:
    oauth = init_oauth()
except Exception as e:
    logger.error(f"Failed to initialize OAuth: {e}")
    oauth = None

@oauth_router.get("/google/login")
async def google_login(request: Request):
    """Initiate Google OAuth login"""
    
    if not oauth:
        raise HTTPException(
            status_code=500, 
            detail="OAuth not configured properly"
        )
    
    try:
        redirect_uri = os.getenv('GOOGLE_REDIRECT_URI')
        if not redirect_uri:
            raise HTTPException(
                status_code=500,
                detail="Google redirect URI not configured"
            )
        
        # Generate authorization URL
        return await oauth.google.authorize_redirect(request, redirect_uri)
        
    except Exception as e:
        logger.error(f"Error initiating Google OAuth: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to initiate Google login"
        )

@oauth_router.get("/google/callback")
async def google_callback(request: Request, db: Session = Depends(get_db)):
    """Handle Google OAuth callback with mobile collection flow"""
    
    if not oauth:
        frontend_url = os.getenv('FRONTEND_URL', 'http://localhost:3000')
        return RedirectResponse(f"{frontend_url}/login?error=oauth_not_configured")
    
    try:
        # Get access token from Google
        token = await oauth.google.authorize_access_token(request)
        if not token:
            frontend_url = os.getenv('FRONTEND_URL', 'http://localhost:3000')
            return RedirectResponse(f"{frontend_url}/login?error=authentication_failed")
        
        # Get user info from Google
        user_info = token.get('userinfo')
        if not user_info or 'email' not in user_info:
            frontend_url = os.getenv('FRONTEND_URL', 'http://localhost:3000')
            return RedirectResponse(f"{frontend_url}/login?error=user_info_failed")
        
        # Find or create user
        user = db.query(User).filter(User.email == user_info['email']).first()
        
        is_new_user = False
        if user:
            # Update existing user info
            user.last_login = get_ist_now()
            if not user.oauth_provider:
                user.oauth_provider = 'google'
                user.oauth_id = user_info.get('sub')
                user.is_oauth_user = True
        else:
            # Create new OAuth user
            is_new_user = True
            user = User(
                username=user_info['email'],
                email=user_info['email'],
                name=user_info.get('name', ''),
                oauth_provider='google',
                oauth_id=user_info.get('sub'),
                is_oauth_user=True,
                password_hash=None,
                mobile_number=None,  # Will be collected later
                last_login=get_ist_now()
            )
            db.add(user)
            db.commit()
            db.refresh(user)
        
        # Generate JWT token
        jwt_secret = os.getenv('JWT_SECRET')
        access_token = jwt.encode({
            'user_id': user.id,
            'username': user.username,
            'exp': get_ist_now() + timedelta(days=7),
            'iat': get_ist_now(),
            'oauth': True
        }, jwt_secret, algorithm="HS256")
        
        # Prepare auth data for frontend
        auth_data = {
            'token': access_token,
            'user_id': user.id,
            'username': user.username,
            'email': user.email,
            'name': user.name or user_info.get('name', ''),
            'oauth_provider': 'google',
            'login_method': 'oauth',
            'is_oauth_user': True,
            'is_new_user': str(is_new_user).lower(),
            'needs_mobile': str(not bool(user.mobile_number)).lower()  # Key addition
        }
        
        # Redirect to frontend with auth data
        frontend_url = os.getenv('FRONTEND_URL', 'http://localhost:3000')
        auth_params = urlencode(auth_data)
        redirect_url = f"{frontend_url}/auth/callback?{auth_params}"
        
        logger.info(f"Google OAuth successful for user: {user_info['email']}, needs_mobile: {not bool(user.mobile_number)}")
        return RedirectResponse(redirect_url)
        
    except Exception as e:
        logger.error(f"Error in Google OAuth callback: {e}")
        frontend_url = os.getenv('FRONTEND_URL', 'http://localhost:3000')
        error_params = urlencode({
            'error': 'authentication_failed', 
            'message': str(e)
        })
        return RedirectResponse(f"{frontend_url}/login?{error_params}")
    
    
@oauth_router.get("/oauth/status")
async def oauth_status():
    """Check OAuth configuration status"""
    try:
        client_id = os.getenv('GOOGLE_OAUTH_CLIENT_ID')
        client_secret = os.getenv('GOOGLE_OAUTH_CLIENT_SECRET')
        
        return {
            'status': 'success',
            'oauth_configured': bool(client_id and client_secret and oauth),
            'google_available': bool(oauth)
        }
    except Exception as e:
        logger.error(f"Error checking OAuth status: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to check OAuth status"
        )

@oauth_router.get("/user/profile")
async def get_oauth_user_profile(
    token: str,
    db: Session = Depends(get_db)
):
    """Get user profile from OAuth token (for frontend use)"""
    try:
        jwt_secret = os.getenv('JWT_SECRET')
        payload = jwt.decode(token, jwt_secret, algorithms=["HS256"])
        user_id = payload.get('user_id')
        
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(
                status_code=404,
                detail="User not found"
            )
        
        return {
            'id': user.id,
            'username': user.username,
            'email': user.email,
            'name': user.name,
            'oauth_provider': user.oauth_provider,
            'is_oauth_user': user.is_oauth_user,
            'oauth_login': payload.get('oauth', False)
        }
        
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=401,
            detail="Token has expired"
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=401,
            detail="Invalid token"
        )
    except Exception as e:
        logger.error(f"Error getting user profile: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to get user profile"
        )