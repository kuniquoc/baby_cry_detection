from firebase_admin import messaging
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Notification templates
notification_templates = {
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

def send_fcm_notification(token, title, body, data=None):
    """
    Send a push notification using Firebase Cloud Messaging
    
    Parameters:
    ----------
    token : str
        The FCM registration token for the target device
    title : str
        The notification title
    body : str
        The notification body message
    data : dict, optional
        Additional data to send with the notification
        
    Returns:
    -------
    bool
        True if the message was sent successfully, False otherwise
    """
    try:
        # Create message
        message = messaging.Message(
            notification=messaging.Notification(
                title=title,
                body=body,
            ),
            android=messaging.AndroidConfig(
                priority='high',
                notification=messaging.AndroidNotification(
                    priority='high',
                    sound='notification.mp3',
                    channel_id='default',
                    default_sound=False,
                ),
                direct_boot_ok=True,
            ),
            data=data or {},
            token=token,
        )
            
        # Send message
        response = messaging.send(message)
        logger.info(f"Successfully sent FCM notification: {response}")
        return True
        
    except Exception as e:
        logger.error(f"Error sending FCM notification: {str(e)}")
        return False

async def send_crying_notification_fcm(deviceId, tokens_with_info, duration=None):
    """
    Send a crying detection notification via FCM
    
    Parameters:
    ----------
    device_id : str
        The device ID to send notification for
    duration : float, optional
        The duration of crying in seconds
        
    Returns:
    -------
    bool
        True if all notifications were sent successfully, False otherwise
    """
    try:
            
        current_time = int(time.time())
        duration_str = str(int(duration)) if duration else ""
        
        # Prepare data payload according to schema
        data = {
            "type": "Crying",
            "time": str(current_time),
            "duration": duration_str,
            "device_id": deviceId,
        }
        
        success = True
        # Send to each token with appropriate language
        for token, language, device_name in tokens_with_info:
            # Get notification template for language (default to English)
            template = notification_templates['crying'].get(language, notification_templates['crying']['en'])
            
            # Format notification
            title = template['title'].format(device_name=device_name)
            body = template['body'].format(duration=duration_str) if duration else \
                   "Baby is crying" if language == 'en' else "Bé đang khóc"
            
            # Send notification
            if not send_fcm_notification(token, title, body, data):
                success = False
                
        return success
        
    except Exception as e:
        logger.error(f"Error sending crying FCM notification: {str(e)}")
        return False
