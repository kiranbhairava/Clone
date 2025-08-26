import logging
import json
import time
from datetime import datetime
from pathlib import Path
import pytz

# IST timezone configuration
IST = pytz.timezone('Asia/Kolkata')

# Create logs directory
Path("logs").mkdir(exist_ok=True)

# Simple token counter (rough estimation)
def count_tokens(text: str) -> int:
    """Simple token estimation: ~4 chars per token"""
    return len(text) // 4 if text else 0

def get_ist_now():
    """Get current datetime in IST"""
    return datetime.now(IST)

def get_ist_timestamp_str():
    """Get IST timestamp as formatted string"""
    return get_ist_now().strftime('%Y-%m-%d %H:%M:%S IST')

class ISTFormatter(logging.Formatter):
    """Custom formatter to use IST timezone"""
    
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, IST)
        if datefmt:
            s = dt.strftime(datefmt)
        else:
            s = dt.strftime('%Y-%m-%d %H:%M:%S IST')
        return s

# Setup logger
def setup_logger():
    """Setup simple logging with IST timestamps"""
    
    # Main logger
    logger = logging.getLogger("chat_app")
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # File handler
    file_handler = logging.FileHandler("logs/app.log")
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # IST Formatter
    formatter = ISTFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Stats handler (JSON format) - separate handler for stats
    stats_handler = logging.FileHandler("logs/stats.jsonl")
    stats_handler.setLevel(logging.INFO)
    
    return logger, stats_handler

# Global logger
app_logger, stats_handler = setup_logger()

def log_chat_interaction(user_id: int, character_name: str, user_input: str, 
                        ai_response: str, response_time_ms: float, language: str = "english"):
    """Log a chat interaction with token stats using IST timestamps"""
    
    input_tokens = count_tokens(user_input)
    output_tokens = count_tokens(ai_response)
    total_tokens = input_tokens + output_tokens
    estimated_cost = total_tokens * 0.00001  # Rough estimate
    
    # Get IST timestamp
    ist_now = get_ist_now()
    
    # Log to main log with IST
    app_logger.info(f"Chat - User:{user_id} Character:{character_name} "
                   f"Lang:{language} Time:{response_time_ms:.0f}ms "
                   f"Tokens:{input_tokens}+{output_tokens}={total_tokens}")
    
    # Log stats as JSON with IST timestamp
    stats_data = {
        "timestamp": ist_now.isoformat(),
        "timezone": "Asia/Kolkata",
        "type": "chat",
        "user_id": user_id,
        "character_name": character_name,
        "language": language,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "response_time_ms": response_time_ms,
        "estimated_cost": estimated_cost,
        "user_input_length": len(user_input),
        "ai_response_length": len(ai_response)
    }
    
    # Write to stats file
    with open("logs/stats.jsonl", "a", encoding='utf-8') as f:
        f.write(json.dumps(stats_data, ensure_ascii=False) + "\n")

def log_user_activity(user_id: int, activity: str, details: dict = None):
    """Log user activity with IST timestamps"""
    ist_now = get_ist_now()
    details = details or {}
    
    # Add IST timestamp to details
    details["timestamp"] = ist_now.isoformat()
    details["timezone"] = "Asia/Kolkata"
    
    app_logger.info(f"User Activity - ID:{user_id} Action:{activity} Details:{details}")
    
    # Also write detailed activity to separate file
    activity_data = {
        "timestamp": ist_now.isoformat(),
        "timezone": "Asia/Kolkata",
        "type": "user_activity",
        "user_id": user_id,
        "activity": activity,
        "details": details
    }
    
    with open("logs/user_activity.jsonl", "a", encoding='utf-8') as f:
        f.write(json.dumps(activity_data, ensure_ascii=False) + "\n")

def log_error(error: Exception, context: dict = None):
    """Log errors with IST timestamps"""
    ist_now = get_ist_now()
    context = context or {}
    
    # Add IST timestamp to context
    context["timestamp"] = ist_now.isoformat()
    context["timezone"] = "Asia/Kolkata"
    context["error_type"] = type(error).__name__
    
    app_logger.error(f"Error: {str(error)} Context:{context}", exc_info=True)
    
    # Write error details to separate file
    error_data = {
        "timestamp": ist_now.isoformat(),
        "timezone": "Asia/Kolkata",
        "type": "error",
        "error_message": str(error),
        "error_type": type(error).__name__,
        "context": context
    }
    
    with open("logs/errors.jsonl", "a", encoding='utf-8') as f:
        f.write(json.dumps(error_data, ensure_ascii=False) + "\n")

def log_api_endpoint(endpoint: str, user_id: int, response_time_ms: float, success: bool = True):
    """Log API endpoint usage with IST timestamps"""
    ist_now = get_ist_now()
    status = "SUCCESS" if success else "ERROR"
    
    app_logger.info(f"API - {endpoint} User:{user_id} Time:{response_time_ms:.0f}ms Status:{status}")
    
    # Write API usage stats
    api_data = {
        "timestamp": ist_now.isoformat(),
        "timezone": "Asia/Kolkata",
        "type": "api_call",
        "endpoint": endpoint,
        "user_id": user_id,
        "response_time_ms": response_time_ms,
        "success": success,
        "status": status
    }
    
    with open("logs/api_usage.jsonl", "a", encoding='utf-8') as f:
        f.write(json.dumps(api_data, ensure_ascii=False) + "\n")

def log_security_event(event_type: str, details: dict):
    """Log security-related events with IST timestamps"""
    ist_now = get_ist_now()
    
    security_data = {
        "timestamp": ist_now.isoformat(),
        "timezone": "Asia/Kolkata",
        "type": "security",
        "event": event_type,
        "details": details
    }
    
    # Write to security-specific log file
    with open("logs/security.jsonl", "a", encoding='utf-8') as f:
        f.write(json.dumps(security_data, ensure_ascii=False) + "\n")
    
    # Also log to main logger
    app_logger.info(f"SECURITY: {event_type} - {details}")

def log_rate_limit_event(endpoint: str, user_identifier: str, limit_type: str):
    """Log rate limit violations with IST timestamps"""
    ist_now = get_ist_now()
    
    rate_limit_data = {
        "timestamp": ist_now.isoformat(),
        "timezone": "Asia/Kolkata",
        "type": "rate_limit",
        "endpoint": endpoint,
        "user_identifier": user_identifier,
        "limit_type": limit_type
    }
    
    with open("logs/rate_limits.jsonl", "a", encoding='utf-8') as f:
        f.write(json.dumps(rate_limit_data, ensure_ascii=False) + "\n")
    
    app_logger.warning(f"RATE LIMIT: {endpoint} - {user_identifier} ({limit_type})")

# Simple decorator for timing with IST
def time_it(func):
    """Simple timing decorator with IST logging"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        ist_start = get_ist_now()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000
            app_logger.info(f"Function {func.__name__} took {duration_ms:.2f}ms at {ist_start.strftime('%H:%M:%S IST')}")
    return wrapper

# System startup logging
def log_system_startup():
    """Log system startup with IST timestamp"""
    ist_now = get_ist_now()
    startup_data = {
        "timestamp": ist_now.isoformat(),
        "timezone": "Asia/Kolkata",
        "type": "system",
        "event": "startup",
        "message": "Character Chat API starting up"
    }
    
    with open("logs/system.jsonl", "a", encoding='utf-8') as f:
        f.write(json.dumps(startup_data, ensure_ascii=False) + "\n")
    
    app_logger.info(f"=== CHARACTER CHAT API STARTUP at {get_ist_timestamp_str()} ===")

def log_system_shutdown():
    """Log system shutdown with IST timestamp"""
    ist_now = get_ist_now()
    shutdown_data = {
        "timestamp": ist_now.isoformat(),
        "timezone": "Asia/Kolkata",
        "type": "system",
        "event": "shutdown",
        "message": "Character Chat API shutting down"
    }
    
    with open("logs/system.jsonl", "a", encoding='utf-8') as f:
        f.write(json.dumps(shutdown_data, ensure_ascii=False) + "\n")
    
    app_logger.info(f"=== CHARACTER CHAT API SHUTDOWN at {get_ist_timestamp_str()} ===")

# Initialize logging system
log_system_startup()