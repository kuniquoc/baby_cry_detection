import firebase_admin
from firebase_admin import credentials, firestore
import os
import logging
import base64
import json
from typing import Dict, List, Any, Optional, Tuple, Callable
from fcm_service import send_crying_notification_fcm
from dotenv import load_dotenv

from utils.date_utils import convert_utc_timestamp_to_vn_datetime

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name%s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_CRY_THRESHOLD = 10  # seconds
EVENT_TYPES = {
    'CRYING': 'crying',
    'NO_CRYING': 'nocrying'
}

# Firebase state
firebase_initialized: bool = False
firestore_client: Optional[firestore.Client] = None

class FirebaseInitError(Exception):
    """Custom exception for Firebase initialization errors"""
    pass

def initialize_firebase() -> bool:
    """Initialize Firebase connection if not already initialized"""
    global firebase_initialized, firestore_client
    
    if firebase_initialized:
        return True
    
    try:
        # Get base64 encoded credentials from environment variable
        firebase_creds_base64 = os.getenv('FIREBASE_CREDENTIALS_BASE64')
        if not firebase_creds_base64:
            raise FirebaseInitError("FIREBASE_CREDENTIALS_BASE64 environment variable not found")
            
        # Decode base64 credentials
        try:
            creds_json = base64.b64decode(firebase_creds_base64).decode('utf-8')
            creds_dict = json.loads(creds_json)
        except Exception as e:
            raise FirebaseInitError(f"Failed to decode Firebase credentials: {str(e)}")
            
        cred = credentials.Certificate(creds_dict)
        firebase_admin.initialize_app(cred)
        firestore_client = firestore.client()
        
        firebase_initialized = True
        logger.info("Firebase successfully initialized using base64 credentials")
        return True
        
    except Exception as e:
        logger.error(f"Error initializing Firebase: {str(e)}")
        return False

def get_device_settings(device_ref: firestore.DocumentReference) -> Tuple[int, bool]:
    """Get device settings from Firestore"""
    try:
        device_doc = device_ref.get()
        if not device_doc.exists:
            logger.error("Device document does not exist")
            return DEFAULT_CRY_THRESHOLD, False
            
        device_data = device_doc.to_dict()
        if not device_data:
            logger.error("Device document is empty")
            return DEFAULT_CRY_THRESHOLD, False
            
        cry_threshold = device_data.get('cryingThreshold', DEFAULT_CRY_THRESHOLD)
        return cry_threshold, True
        
    except Exception as e:
        logger.error(f"Error getting device settings: {str(e)}")
        return DEFAULT_CRY_THRESHOLD, False

def create_notification_doc(timestamp: int, duration: float) -> Dict[str, Any]:
    """Create notification document data"""
    dt = convert_utc_timestamp_to_vn_datetime(timestamp)
    return {
        'type': EVENT_TYPES['CRYING'],
        'time': dt,  # Firestore will automatically convert datetime to Timestamp
        'duration': duration,
        'imageUrl': ''  # Required by schema
    }

def should_create_notification(last_notification: Optional[Dict[str, Any]], 
                            current_timestamp: float,
                            cry_threshold: float,
                            duration: float) -> bool:
    """Determine if a new notification should be created"""
    if not last_notification:
        return duration >= cry_threshold
        
    last_notification_time = last_notification.get('time').timestamp()
    time_since_last_notification = current_timestamp - last_notification_time
    
    return time_since_last_notification >= cry_threshold and duration >= cry_threshold

async def send_cry_notification(notification_data: Dict[str, Any]) -> bool:
    """Send cry detection notification to Firebase using real-time data"""
    if not notification_data:
        logger.error("Notification data is None or empty")
        return False
        
    if not initialize_firebase() or not firestore_client:
        logger.error("Failed to initialize Firebase")
        return False
    
    try:
        device_id = notification_data.get('deviceId')
        current_timestamp = notification_data.get('timestamp')
        
        if not device_id or not current_timestamp:
            logger.error("DeviceId and timestamp are required")
            return False
            
        # Validate device exists in Firestore
        device_ref = firestore_client.collection('devices').document(device_id)
        if not device_ref.get().exists:
            logger.error(f"Device {device_id} does not exist in Firestore")
            return False
            
        # Get Firestore references
        events_ref = device_ref.collection('events')
        notifications_ref = device_ref.collection('notifications')
        
        # Get device settings
        cry_threshold, settings_success = get_device_settings(device_ref)
        if not settings_success:
            logger.warning(f"Using default cry threshold for device {device_id}")
            
        # Get latest event
        latest_events = (events_ref
                        .where(field_path="type", op_string="in", value=[EVENT_TYPES['CRYING'], EVENT_TYPES['NO_CRYING']])
                        .order_by('time', direction=firestore.Query.DESCENDING)
                        .limit(1)
                        .stream())
        
        last_event = next(({"id": event.id, **event.to_dict()} for event in latest_events), None)
        
        # Create new crying event if last event was NO_CRYING or no event exists
        if not last_event or last_event.get('type') == EVENT_TYPES['NO_CRYING']:
            new_event = {
                'type': EVENT_TYPES['CRYING'],
                'time': convert_utc_timestamp_to_vn_datetime(current_timestamp)
            }
            events_ref.add(new_event)
            event_time = convert_utc_timestamp_to_vn_datetime(current_timestamp)
            logger.info(f"Added new Crying event at {event_time} for device {device_id}")
            return True
            
        # If last event was CRYING, check if we should create a notification
        if last_event and last_event.get('type') == EVENT_TYPES['CRYING']:
            event_time = last_event.get('time')
            if not event_time:
                logger.error(f"Missing timestamp in last crying event for device {device_id}")
                return False
                
            last_crying_time = event_time.timestamp()
            duration = current_timestamp - last_crying_time
            
            # Get latest notification
            latest_notifications = (notifications_ref
                                  .where(field_path="type", op_string="==", value=EVENT_TYPES['CRYING'])
                                  .order_by('time', direction=firestore.Query.DESCENDING)
                                  .limit(1)
                                  .stream())
            
            last_notification = next(({"id": notif.id, **notif.to_dict()} for notif in latest_notifications), None)
            
            # Check if we should create a new notification
            if should_create_notification(last_notification, current_timestamp, cry_threshold, duration):
                notification_doc_data = {
                    'type': EVENT_TYPES['CRYING'],
                    'time': convert_utc_timestamp_to_vn_datetime(current_timestamp),
                    'duration': duration,
                    'imageUrl': ''  # Required by schema
                }
                notifications_ref.add(notification_doc_data)
                
                tokens_with_info, deviceId = await get_fcm_tokens_for_device(device_id)
                if tokens_with_info:
                    await send_crying_notification_fcm(deviceId, tokens_with_info, duration)
                else:
                    logger.warning(f"No FCM tokens found for device {device_id}")
                logger.info(f"Created Crying notification with duration {duration:.2f}s for device {device_id}")
        
        return True
            
    except Exception as e:
        logger.error(f"Error processing notification: {str(e)}")
        return False

async def send_nocry_notification(data: Dict[str, Any]) -> bool:
    """Update Firebase when baby has stopped crying"""
    if not data or not all(key in data for key in ['deviceId', 'timestamp', 'lastCryTimestamp']):
        logger.error("Invalid no-cry notification data")
        return False
        
    if not initialize_firebase() or not firestore_client:
        logger.error("Failed to initialize Firebase")
        return False
    
    try:
        device_id = data['deviceId']
        current_timestamp = float(data['timestamp'])  # Keep as float to preserve precision
        
        events_ref = firestore_client.collection('devices').document(device_id).collection('events')
        
        # Get most recent event
        latest_events = (events_ref
                        .where(field_path="type", op_string="in", value=[EVENT_TYPES['CRYING'], EVENT_TYPES['NO_CRYING']])
                        .order_by('time', direction=firestore.Query.DESCENDING)
                        .limit(1)
                        .stream())
        
        last_event = next(({"id": event.id, **event.to_dict()} for event in latest_events), None)
            
        # Add NoCrying event regardless of last event type if we have a valid lastCryTimestamp
        # This ensures no-cry events are created when needed
        if data.get('lastCryTimestamp') is not None:
            # Only add if last event was NOT already a no-crying event at the same time
            should_add_event = True
            if last_event and last_event.get('type') == EVENT_TYPES['NO_CRYING']:
                # Check if the last no-cry event was for the same cry session
                last_event_timestamp = last_event.get('time').timestamp() if last_event.get('time') else 0
                # If last no-cry event was very recent (within 2 seconds), skip to avoid duplicates
                if abs(current_timestamp - last_event_timestamp) < 2:
                    should_add_event = False
                    logger.debug(f"Skipping duplicate no-cry event for device {device_id}")
            
            if should_add_event:
                events_ref.add({
                    'type': EVENT_TYPES['NO_CRYING'],
                    'time': convert_utc_timestamp_to_vn_datetime(current_timestamp)  # Convert to datetime
                })
                
                logger.info(f"Added NoCrying event at {convert_utc_timestamp_to_vn_datetime(data['timestamp'])} for device {device_id}")
        else:
            logger.warning(f"No lastCryTimestamp provided for device {device_id}, skipping no-cry event")
            
        return True
            
    except Exception as e:
        logger.error(f"Error sending no-cry event: {str(e)}")
        return False

async def get_fcm_tokens_for_device(device_id: str) -> Tuple[List[Tuple[str, str, str]], str]:
    """Get FCM tokens for a device's connections"""
    try:
        if not initialize_firebase() or not firestore_client:
            logger.error("Failed to initialize Firebase")
            return [], device_id
            
        fcm_tokens: List[Tuple[str, str, str]] = []
        
        connections = firestore_client.collection('connections') \
                                   .where(field_path='deviceId', op_string='==', value=device_id) \
                                   .stream()
                                   
        for connection in connections:
            conn_data = connection.to_dict()
            user_id = conn_data.get('userId')
            conn_name = conn_data.get('name', '')
            
            if user_id:
                user_doc = firestore_client.collection('users').document(user_id).get()
                if user_doc.exists:
                    user_data = user_doc.to_dict()
                    fcm_tokens_data = user_data.get('fcmTokens', [])
                    tokens_with_info = [
                        (token, user_data.get('language', 'en'), conn_name)
                        for token in fcm_tokens_data
                    ]
                    fcm_tokens.extend(tokens_with_info)
        
        return fcm_tokens, device_id
        
    except Exception as e:
        logger.error(f"Error getting FCM tokens for device {device_id}: {str(e)}")
        return [], device_id