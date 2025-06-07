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
        if not token or not token.strip():
            logger.warning("Empty or invalid FCM token provided - skipping")
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
        logger.warning(f"FCM token is unregistered - skipping token: {token[:20]}...")
        return False
    except messaging.SenderIdMismatchError:
        logger.warning(f"Sender ID mismatch in FCM token - skipping token: {token[:20]}...")
        return False
    except messaging.InvalidArgumentError:
        logger.warning(f"Invalid FCM token format - skipping token: {token[:20]}...")
        return False
    except messaging.QuotaExceededError:
        logger.error("FCM quota exceeded - this affects all tokens")
        return False
    except Exception as e:
        logger.warning(f"Error sending FCM notification (skipping token): {str(e)}")
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
        
        success_count = 0
        skipped_count = 0
        total_tokens = len(tokens_with_info)
        
        logger.info(f"Attempting to send notifications to {total_tokens} FCM tokens")
        
        for token, language, device_name in tokens_with_info:
            # Skip obviously invalid tokens
            if not token or not token.strip():
                skipped_count += 1
                continue
                
            template = NOTIFICATION_TEMPLATES['crying'].get(language, NOTIFICATION_TEMPLATES['crying']['en'])
            
            title = template['title'].format(device_name=device_name)
            body = template['body'].format(duration=duration_str) if duration else DEFAULT_MESSAGES.get(language, DEFAULT_MESSAGES['en'])
            
            if send_fcm_notification(token, title, body, data):
                success_count += 1
            else:
                skipped_count += 1
        
        logger.info(f"FCM notification results: {success_count} sent, {skipped_count} skipped out of {total_tokens} total")
        
        # Return True if at least one notification was sent successfully
        return success_count > 0
        
    except Exception as e:
        logger.error(f"Error sending crying FCM notification: {str(e)}")
        return False
