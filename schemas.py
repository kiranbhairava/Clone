from pydantic import BaseModel, field_validator, EmailStr
from typing import Optional, List
from datetime import datetime
import re

# User schemas
class UserCreate(BaseModel):
    username: str
    email: str  # Now required
    mobile_number: str  # New required field
    password: str
    
    @field_validator('email')
    @classmethod
    def validate_email(cls, v):
        if not v or not v.strip():
            raise ValueError('Email is required')
        # Basic email validation
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, v.strip()):
            raise ValueError('Invalid email format')
        return v.strip().lower()
    
    @field_validator('mobile_number')
    @classmethod
    def validate_mobile_number(cls, v):
        if not v or not v.strip():
            raise ValueError('Mobile number is required')
        # Remove spaces and common separators
        mobile = re.sub(r'[\s\-\(\)]', '', v.strip())
        # Check if it's a valid mobile number (10 digits, optionally starting with +91)
        if mobile.startswith('+91'):
            mobile = mobile[3:]
        elif mobile.startswith('91') and len(mobile) == 12:
            mobile = mobile[2:]
        
        if not re.match(r'^[6-9]\d{9}$', mobile):
            raise ValueError('Mobile number must be a valid 10-digit Indian mobile number')
        return mobile
    
    @field_validator('username')
    @classmethod
    def validate_username(cls, v):
        if not v or not v.strip():
            raise ValueError('Username is required')
        username = v.strip()
        if len(username) < 3:
            raise ValueError('Username must be at least 3 characters long')
        if len(username) > 50:
            raise ValueError('Username must be less than 50 characters')
        # Allow alphanumeric, underscore, hyphen, and dot
        if not re.match(r'^[a-zA-Z0-9._-]+$', username):
            raise ValueError('Username can only contain letters, numbers, dots, underscores, and hyphens')
        return username
    
    @field_validator('password')
    @classmethod
    def validate_password(cls, v):
        if not v:
            raise ValueError('Password is required')
        if len(v) < 6:
            raise ValueError('Password must be at least 6 characters long')
        if len(v) > 100:
            raise ValueError('Password is too long (max 100 characters)')
        return v
    
class MobileUpdateRequest(BaseModel):
    mobile_number: str
    
    @field_validator('mobile_number')
    @classmethod
    def validate_mobile_number(cls, v):
        if not v or not v.strip():
            raise ValueError('Mobile number is required')
        # Remove spaces and common separators
        mobile = re.sub(r'[\s\-\(\)]', '', v.strip())
        # Check if it's a valid mobile number (10 digits, optionally starting with +91)
        if mobile.startswith('+91'):
            mobile = mobile[3:]
        elif mobile.startswith('91') and len(mobile) == 12:
            mobile = mobile[2:]
        
        if not re.match(r'^[6-9]\d{9}$', mobile):
            raise ValueError('Mobile number must be a valid 10-digit Indian mobile number')
        return mobile

class UserLogin(BaseModel):
    username: str  # Can be username, email, or mobile number
    password: str
    
    @field_validator('username')
    @classmethod
    def validate_username(cls, v):
        if not v or not v.strip():
            raise ValueError('Username/Email/Mobile is required')
        return v.strip()
    
    @field_validator('password')
    @classmethod
    def validate_password(cls, v):
        if not v:
            raise ValueError('Password is required')
        return v

class UserResponse(BaseModel):
    id: int
    username: str
    email: Optional[str] = None
    mobile_number: Optional[str] = None
    created_at: datetime
    
    class Config:
        from_attributes = True

# Character schemas
class CharacterBase(BaseModel):
    name: str
    description: str
    img: Optional[str] = None
    native_language: Optional[str] = "english"
    is_multilingual: Optional[bool] = True
    category: Optional[str] = "entertainment_arts"

class CharacterCreate(CharacterBase):
    # Character creation form fields
    age_range: Optional[str] = None
    setting: Optional[str] = None
    profession: Optional[str] = None
    cultural_background: Optional[str] = None
    known_for: List[str] = []
    speaking_style: str = "casual_friendly"
    common_phrases: Optional[str] = None
    response_length: str = "varies"
    communication_quirks: List[str] = []
    excited_about: List[str] = []
    frustrated_by: List[str] = []
    deep_values: Optional[str] = None
    good_at: List[str] = []
    not_good_at: List[str] = []
    love_discussing: str = "life and personal growth"
    avoid_discussing: Optional[str] = None
    personality_optimistic: bool = True
    personality_patient: bool = True
    personality_serious: bool = False
    personality_introverted: Optional[bool] = None
    character_flaws: List[str] = []
    unique_trait: str = "your genuine care for others"
    disagreement_style: str = "listen_understand"
    emotional_triggers: Optional[str] = None
    help_style: str = "ask_questions"
    safety_level: str = "respectful_discussion"
    content_tone: str = "warm and helpful"
    restricted_topics: Optional[str] = None

class CharacterResponse(CharacterBase):
    id: int
    created_at: datetime
    
    @field_validator('native_language', mode='before')
    @classmethod
    def validate_native_language(cls, v):
        return v if v is not None else "english"
    
    @field_validator('is_multilingual', mode='before')
    @classmethod
    def validate_is_multilingual(cls, v):
        return v if v is not None else True
    
    @field_validator('category', mode='before')
    @classmethod
    def validate_category(cls, v):
        return v if v is not None else "entertainment_arts"
    
    class Config:
        from_attributes = True

class CharacterListResponse(BaseModel):
    id: int
    name: str
    img: Optional[str]
    description: str
    native_language: Optional[str] = "english"
    is_multilingual: bool = True
    category: Optional[str] = "entertainment_arts"
    
    @field_validator('native_language', mode='before')
    @classmethod
    def validate_native_language(cls, v):
        return v if v is not None else "english"
    
    @field_validator('is_multilingual', mode='before')
    @classmethod
    def validate_is_multilingual(cls, v):
        return v if v is not None else True
    
    @field_validator('category', mode='before')
    @classmethod
    def validate_category(cls, v):
        return v if v is not None else "entertainment_arts"
    
    class Config:
        from_attributes = True

# Chat schemas
class ChatRequest(BaseModel):
    character_name: str
    user_input: str
    new_session: bool = False
    language: Optional[str] = None

class ChatMessage(BaseModel):
    role: str
    content: str
    language: str = "english"
    timestamp: Optional[str] = None

class ChatResponse(BaseModel):
    reply: str
    chat_history: List[ChatMessage]
    session_id: int
    input_language: str
    response_language: str
    character_native_language: str

# Session schemas
class SessionResponse(BaseModel):
    session_id: int
    character: str
    primary_language: str
    created_at: str

class SessionMessagesResponse(BaseModel):
    character: str
    primary_language: str
    chat_history: List[ChatMessage]
    created_at: str
    status: str

# Pagination schemas
class PaginationInfo(BaseModel):
    page: int
    per_page: int
    total_pages: int
    next_url: Optional[str] = None
    prev_url: Optional[str] = None
    total_count: int
    has_next: bool
    has_prev: bool

class CharacterListPaginated(BaseModel):
    characters: List[CharacterListResponse]
    pagination: PaginationInfo
    status: str

# Search Response
class SearchResponse(BaseModel):
    results: List[CharacterListResponse]
    search_term: str
    page: int
    per_page: int
    total_pages: int
    total_count: int
    next_url: Optional[str] = None
    prev_url: Optional[str] = None
    has_next: bool
    has_prev: bool

# Language schemas
class LanguageInfo(BaseModel):
    code: str
    name: str
    native_name: str
    iso_code: str

class LanguagesResponse(BaseModel):
    languages: List[LanguageInfo]
    status: str

# Category schemas
class CategoryResponse(BaseModel):
    categories: dict
    status: str

class CategoryCharactersResponse(BaseModel):
    count: int
    characters: List[CharacterListResponse]
    category_name: str
    status: str

# Form options schemas
class FormOption(BaseModel):
    key: str
    label: str
    example: Optional[str] = None
    description: Optional[str] = None

class FormOptionsResponse(BaseModel):
    speaking_styles: List[FormOption]
    response_lengths: List[FormOption]
    help_styles: List[FormOption]
    excitement_topics: List[str]
    frustration_topics: List[str]
    communication_quirks: List[str]
    character_flaws: List[str]
    disagreement_styles: List[FormOption]
    safety_levels: List[FormOption]
    age_ranges: List[str]
    settings: List[str]
    supported_languages: List[LanguageInfo]

# Generic response schemas
class SuccessResponse(BaseModel):
    status: str = "success"
    message: str

class ErrorResponse(BaseModel):
    error: str

# Health Response
class HealthResponse(BaseModel):
    status: str
    multilingual_support: bool
    migration_support: bool = True
    version: str