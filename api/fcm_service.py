from firebase_admin import messaging
import logging
import time
from typing import Dict, List, Tuple, Optional, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Constants
ANDROID_NOTIFICATION_CHANNEL = 'babycare-alerts'
ANDROID_PRIORITY = 'high'

# Notification templates
NOTIFICATION_TEMPLATES = {
    'crying': {
        'en': {
            'title': '[{device_name}] Crying alert',
            'body': 'Baby is crying, continuously for {duration} seconds'
        },
        'vi': {
            'title': '[{device_name}] Cảnh báo khóc',
            'body': 'Bé đang khóc, đã liên tục trong {duration} giây'
        }
    }
}

DEFAULT_MESSAGES = {
    'en': 'Baby is crying',
    'vi': 'Bé đang khóc'
}

def send_fcm_notification(token: str, title: str, body: str, data: Optional[Dict[str, Any]] = None) -> bool:
    """Send a push notification using Firebase Cloud Messaging"""
    try:
        # Skip empty or invalid tokens
        if not token:  # Basic validation
            logger.warning("Invalid FCM token provided")
            return False

        message = messaging.Message(
            notification=messaging.Notification(title=title, body=body),
            android=messaging.AndroidConfig(
                priority=ANDROID_PRIORITY,
                notification=messaging.AndroidNotification(
                    priority=ANDROID_PRIORITY,
                    channel_id=ANDROID_NOTIFICATION_CHANNEL,
                ),
            ),
            data=data or {},
            token=token,
        )
            
        response = messaging.send(message)
        logger.info(f"Successfully sent FCM notification: {response}")
        return True
        
    except messaging.UnregisteredError:
        # Token is no longer valid
        logger.warning(f"FCM token is no longer valid: {token}")
        return False
    except messaging.SenderIdMismatchError:
        logger.error("Sender ID mismatch in FCM token")
        return False
    except messaging.QuotaExceededError:
        logger.error("FCM quota exceeded")
        return False
    except Exception as e:
        logger.error(f"Error sending FCM notification: {str(e)}")
        return False

async def send_crying_notification_fcm(deviceId: str, tokens_with_info: List[Tuple[str, str, str]], duration: Optional[float] = None) -> bool:
    """Send a crying detection notification via FCM"""
    try:
        current_time = int(time.time())
        duration_str = str(int(duration)) if duration else ""
        
        data = {
            "type": "Crying",
            "time": str(current_time),
            "duration": duration_str,
            "device_id": deviceId,
        }
        
        success = False  # Track if at least one notification was sent
        invalid_tokens = []
        
        for token, language, device_name in tokens_with_info:
            template = NOTIFICATION_TEMPLATES['crying'].get(language, NOTIFICATION_TEMPLATES['crying']['en'])
            
            title = template['title'].format(device_name=device_name)
            body = template['body'].format(duration=duration_str) if duration else DEFAULT_MESSAGES.get(language, DEFAULT_MESSAGES['en'])
            
            if send_fcm_notification(token, title, body, data):
                success = True
            else:
                invalid_tokens.append(token)
        
        if invalid_tokens:
            logger.warning(f"Found {len(invalid_tokens)} invalid FCM tokens that should be cleaned up")
            # TODO: Implement token cleanup logic here
                
        return success
        
    except Exception as e:
        logger.error(f"Error sending crying FCM notification: {str(e)}")
        return False
