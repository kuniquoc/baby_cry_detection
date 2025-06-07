"""
Configuration file for Cry Detection Service
"""

class CryDetectionConfig:
    """Configuration class for cry detection service"""
    
    # Time intervals (in seconds)
    NO_CRY_CHECK_SECONDS = 10          # Time to wait before sending no-cry notification
    PERIODIC_CHECK_INTERVAL = 1        # Interval for periodic checks
      # Consecutive cry detection settings
    REQUIRED_CONSECUTIVE_CRIES = 3      # Number of consecutive cries required for confirmation
    MAX_GAP_BETWEEN_CRIES = 5.0        # Maximum gap between consecutive cries (seconds) 
    LARGE_GAP_RESET_THRESHOLD = 10.0   # Gap threshold to reset timestamps (seconds)
    
    # Memory management
    MAX_TIMESTAMPS_TO_KEEP = 10        # Maximum number of timestamps to keep per device
    
    # Logging levels for different events
    LOG_LEVELS = {
        'cry_detection': 'DEBUG',       # Individual cry detections
        'cry_confirmation': 'INFO',     # Cry confirmations
        'no_cry_events': 'INFO',        # No-cry events
        'cleanup': 'DEBUG',             # Cleanup operations
        'errors': 'ERROR'               # Error events
    }
    
    # WebSocket message types
    WEBSOCKET_MESSAGES = {
        'no_cry_alert': {
            'type': 'alert',
            'message': 'No crying detected'
        },
        'disconnect_alert': {
            'type': 'alert', 
            'message': 'No crying detected (client disconnected)'
        }
    }
    
    # Firebase notification settings
    FIREBASE_SETTINGS = {
        'retry_attempts': 3,            # Number of retry attempts for failed notifications
        'timeout': 10                   # Timeout for Firebase requests (seconds)
    }

    @classmethod
    def get_config_summary(cls) -> str:
        """Get a summary of current configuration"""
        return f"""
Cry Detection Service Configuration:
=====================================
- Required consecutive cries: {cls.REQUIRED_CONSECUTIVE_CRIES}
- Max gap between cries: {cls.MAX_GAP_BETWEEN_CRIES}s
- No-cry check interval: {cls.NO_CRY_CHECK_SECONDS}s
- Large gap reset threshold: {cls.LARGE_GAP_RESET_THRESHOLD}s
- Max timestamps per device: {cls.MAX_TIMESTAMPS_TO_KEEP}
"""
