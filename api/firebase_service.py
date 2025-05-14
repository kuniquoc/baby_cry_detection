import firebase_admin
from firebase_admin import credentials, firestore
import os
import logging
from typing import Dict, List, Any, Optional, Tuple, Callable
from fcm_service import send_crying_notification_fcm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_CRY_THRESHOLD = 10  # seconds
EVENT_TYPES = {
    'CRYING': 'Crying',
    'NO_CRYING': 'NoCrying'
}

# Firebase state
firebase_initialized: bool = False
firestore_client: Optional[firestore.Client] = None
snapshot_callbacks: Dict[str, List[Callable]] = {}  # Store cleanup callbacks for each device
latest_event_docs: Dict[str, Dict] = {}  # Store latest events by device ID
latest_notification_docs: Dict[str, Dict] = {}  # Store latest notifications by device ID

class FirebaseInitError(Exception):
    """Custom exception for Firebase initialization errors"""
    pass

def initialize_firebase() -> bool:
    """Initialize Firebase connection if not already initialized"""
    global firebase_initialized, firestore_client
    
    if firebase_initialized:
        return True
    
    try:
        cred_path = os.path.join(os.path.dirname(__file__), "firebase-credentials.json")
        if not os.path.exists(cred_path):
            raise FirebaseInitError(f"Firebase credentials file not found at: {cred_path}")
            
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred)
        firestore_client = firestore.client()
        
        firebase_initialized = True
        logger.info("Firebase successfully initialized")
        return True
        
    except Exception as e:
        logger.error(f"Error initializing Firebase: {str(e)}")
        return False

def get_device_settings(device_ref: firestore.DocumentReference) -> Tuple[int, bool]:
    """Get device settings from Firestore"""
    try:
        device_doc = device_ref.get()
        if device_doc.exists:
            device_data = device_doc.to_dict()
            cry_threshold = device_data.get('cryingThreshold', DEFAULT_CRY_THRESHOLD)
        else:
            cry_threshold = DEFAULT_CRY_THRESHOLD
            logger.warning(f"Device not found, using default crying threshold of {cry_threshold}s")
        return cry_threshold, True
    except Exception as e:
        logger.error(f"Error retrieving device settings: {str(e)}")
        return DEFAULT_CRY_THRESHOLD, False

def create_notification_doc(timestamp_ms: int, duration: float) -> Dict[str, Any]:
    """Create notification document data"""
    return {
        'type': EVENT_TYPES['CRYING'],
        'time': timestamp_ms,
        'duration': duration,
        'imageUrl': ''  # Required by schema
    }

def start_listening(device_id: str) -> bool:
    """Start real-time listeners for a device's events and notifications"""
    if not initialize_firebase() or not firestore_client:
        logger.error("Failed to initialize Firebase for real-time listening")
        return False
        
    if device_id in snapshot_callbacks:
        logger.warning(f"Listeners already active for device {device_id}")
        return True
        
    try:
        device_ref = firestore_client.collection('devices').document(device_id)
        events_ref = device_ref.collection('events')
        notifications_ref = device_ref.collection('notifications')
        
        # Initialize data with a one-time query
        latest_events = events_ref.where("type", "in", [EVENT_TYPES['CRYING'], EVENT_TYPES['NO_CRYING']]) \
                               .order_by('time', direction=firestore.Query.DESCENDING) \
                               .limit(1) \
                               .get()
        
        if latest_events:
            latest_event = next(({"id": event.id, **event.to_dict()} for event in latest_events), None)
            if latest_event:
                latest_event_docs[device_id] = latest_event
                
        latest_notifs = notifications_ref.where("type", "==", EVENT_TYPES['CRYING']) \
                                     .order_by('time', direction=firestore.Query.DESCENDING) \
                                     .limit(1) \
                                     .get()
                                     
        if latest_notifs:
            latest_notif = next(({"id": notif.id, **notif.to_dict()} for notif in latest_notifs), None)
            if latest_notif:
                latest_notification_docs[device_id] = latest_notif
        
        # Set up event listener
        def on_events_snapshot(event_snapshots, changes, read_time):
            for change in changes:
                if change.type.name in ['ADDED', 'MODIFIED']:
                    doc_data = {"id": change.document.id, **change.document.to_dict()}
                    if doc_data.get('type') in [EVENT_TYPES['CRYING'], EVENT_TYPES['NO_CRYING']]:
                        latest_event_docs[device_id] = doc_data
        
        # Set up notification listener
        def on_notifications_snapshot(notif_snapshots, changes, read_time):
            for change in changes:
                if change.type.name in ['ADDED', 'MODIFIED']:
                    doc_data = {"id": change.document.id, **change.document.to_dict()}
                    if doc_data.get('type') == EVENT_TYPES['CRYING']:
                        latest_notification_docs[device_id] = doc_data
        
        # Register listeners and store unsubscribe functions
        event_watch = events_ref.where("type", "in", [EVENT_TYPES['CRYING'], EVENT_TYPES['NO_CRYING']]) \
                              .order_by('time', direction=firestore.Query.DESCENDING) \
                              .limit(1) \
                              .on_snapshot(on_events_snapshot)
                              
        notif_watch = notifications_ref.where("type", "==", EVENT_TYPES['CRYING']) \
                                    .order_by('time', direction=firestore.Query.DESCENDING) \
                                    .limit(1) \
                                    .on_snapshot(on_notifications_snapshot)
                                    
        snapshot_callbacks[device_id] = [event_watch, notif_watch]
        logger.info(f"Started real-time listeners for device {device_id}")
        return True
        
    except Exception as e:
        logger.error(f"Error setting up real-time listeners for device {device_id}: {str(e)}")
        return False

def stop_listening(device_id: str) -> bool:
    """Stop real-time listeners for a device"""
    try:
        if device_id in snapshot_callbacks:
            # Call all unsubscribe functions
            for unsubscribe in snapshot_callbacks[device_id]:
                unsubscribe()
            
            # Clean up stored data
            del snapshot_callbacks[device_id]
            latest_event_docs.pop(device_id, None)
            latest_notification_docs.pop(device_id, None)
            
            logger.info(f"Stopped real-time listeners for device {device_id}")
            return True
            
        return False
    except Exception as e:
        logger.error(f"Error stopping listeners for device {device_id}: {str(e)}")
        return False

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
            
        current_timestamp_ms = int(current_timestamp * 1000)
        
        # Ensure real-time listeners are active
        if device_id not in snapshot_callbacks:
            if not start_listening(device_id):
                return False
        
        # Get Firestore references
        device_ref = firestore_client.collection('devices').document(device_id)
        events_ref = device_ref.collection('events')
        notifications_ref = device_ref.collection('notifications')
        
        # Get device settings
        cry_threshold, settings_success = get_device_settings(device_ref)
        if not settings_success:
            return False
            
        # Use cached latest event
        last_event = latest_event_docs.get(device_id)
        
        # Handle new crying event
        if not last_event or last_event.get('type') == EVENT_TYPES['NO_CRYING']:
            new_event = {'type': EVENT_TYPES['CRYING'], 'time': current_timestamp_ms}
            events_ref.add(new_event)
            logger.info(f"Added new Crying event at {current_timestamp} for device {device_id}")
            return True
        
        # Process notification logic for existing crying event
        last_crying_time_ms = last_event.get('time', current_timestamp_ms)
        last_crying_time = last_crying_time_ms / 1000
        time_difference = current_timestamp - last_crying_time
        
        # Use cached latest notification
        last_notification = latest_notification_docs.get(device_id)
        
        # Create new notification if needed
        if should_create_notification(last_notification, time_difference, cry_threshold):
            notification_doc_data = create_notification_doc(current_timestamp_ms, time_difference)
            notifications_ref.add(notification_doc_data)
            
            tokens_with_info, deviceId = await get_fcm_tokens_for_device(device_id)
            if tokens_with_info:
                await send_crying_notification_fcm(deviceId, tokens_with_info, time_difference)
            else:
                logger.warning(f"No FCM tokens found for device {device_id}")
                
            logger.info(f"Created Crying notification with duration {time_difference:.2f}s for device {device_id}")
            
        return True
        
    except Exception as e:
        logger.error(f"Error processing notification: {str(e)}")
        return False

def should_create_notification(last_notification: Optional[Dict[str, Any]], 
                            time_difference: float, 
                            cry_threshold: int) -> bool:
    """Determine if a new notification should be created"""
    if not last_notification:
        return time_difference >= cry_threshold
        
    last_notification_time = last_notification.get('time', 0) / 1000
    notification_time_diff = time_difference - last_notification_time
    
    return time_difference >= cry_threshold and notification_time_diff >= cry_threshold

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
        current_timestamp_ms = int(data['timestamp'] * 1000)
        
        events_ref = firestore_client.collection('devices').document(device_id).collection('events')
        
        # Get most recent event
        latest_events = events_ref.where("type", "in", [EVENT_TYPES['CRYING'], EVENT_TYPES['NO_CRYING']]) \
                              .order_by('time', direction=firestore.Query.DESCENDING) \
                              .limit(1) \
                              .stream()
        
        last_event = next(({"id": event.id, **event.to_dict()} for event in latest_events), None)
            
        # Add NoCrying event only if last event was Crying
        if last_event and last_event.get('type') == EVENT_TYPES['CRYING']:
            events_ref.add({
                'type': EVENT_TYPES['NO_CRYING'],
                'time': current_timestamp_ms
            })
            
            logger.info(f"Added NoCrying event at {data['timestamp']} for device {device_id}")
            
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
                                   .where('deviceId', '==', device_id) \
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