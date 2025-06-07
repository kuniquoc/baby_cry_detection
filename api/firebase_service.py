import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud.firestore_v1 import FieldFilter
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
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
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

class DeviceState:
    """Manages the real-time state for each device."""
    def __init__(self):
        self.crying_threshold: Dict[str, int] = {}
        self.last_event: Dict[str, Optional[Dict[str, Any]]] = {}
        self.last_notification: Dict[str, Optional[Dict[str, Any]]] = {} # NEW: Stores last notification
        self.listeners: Dict[str, Callable] = {}

# Global instance to hold all device states and listeners
device_states = DeviceState()

def initialize_firebase() -> bool:
    """Initialize Firebase connection if not already initialized"""
    global firebase_initialized, firestore_client

    if firebase_initialized:
        return True

    try:
        firebase_creds_base64 = os.getenv('FIREBASE_CREDENTIALS_BASE64')
        if not firebase_creds_base64:
            raise FirebaseInitError("FIREBASE_CREDENTIALS_BASE64 environment variable not found")

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

## Quản lý Snapshot Listener

### Hàm Callback Listener mới

def _on_device_settings_snapshot(doc_snapshot, changes, read_time, device_id: str):
    """Callback function for device settings snapshot listener."""
    try:
        # Handle case where doc_snapshot might be a list or other unexpected types
        if isinstance(doc_snapshot, list):
            if len(doc_snapshot) > 0:
                doc_snapshot = doc_snapshot[0]
            else:
                device_states.crying_threshold[device_id] = DEFAULT_CRY_THRESHOLD
                logger.warning(f"Device {device_id} settings not found (empty list). Using default.")
                return
        
        # Check if it's a valid DocumentSnapshot with exists attribute
        if hasattr(doc_snapshot, 'exists'):
            if doc_snapshot.exists:
                data = doc_snapshot.to_dict()
                if data:
                    threshold = data.get('cryingThreshold', DEFAULT_CRY_THRESHOLD)
                    device_states.crying_threshold[device_id] = threshold
                    logger.debug(f"Updated cryingThreshold for {device_id}: {threshold}")
                else:
                    device_states.crying_threshold[device_id] = DEFAULT_CRY_THRESHOLD
                    logger.warning(f"Device {device_id} settings document is empty. Using default.")
            else:
                device_states.crying_threshold[device_id] = DEFAULT_CRY_THRESHOLD
                logger.warning(f"Device {device_id} settings document does not exist. Using default.")
        else:
            # Handle unexpected data types
            logger.error(f"Unexpected snapshot type for device {device_id}: {type(doc_snapshot)}")
            device_states.crying_threshold[device_id] = DEFAULT_CRY_THRESHOLD
            
    except Exception as e:
        logger.error(f"Error in _on_device_settings_snapshot for {device_id}: {str(e)} - Snapshot type: {type(doc_snapshot)}")
        device_states.crying_threshold[device_id] = DEFAULT_CRY_THRESHOLD

def _on_events_snapshot(col_snapshot, changes, read_time, device_id: str):
    """Callback function for events collection snapshot listener."""
    try:
        # Handle case where col_snapshot might be a list or other unexpected types
        if isinstance(col_snapshot, list):
            if len(col_snapshot) > 0:
                # If it's a list of documents, get the first one
                last_doc = col_snapshot[0]
                if hasattr(last_doc, 'id') and hasattr(last_doc, 'to_dict'):
                    device_states.last_event[device_id] = {"id": last_doc.id, **last_doc.to_dict()}
                    logger.debug(f"Updated last_event for {device_id}: Type={device_states.last_event[device_id].get('type')} Time={device_states.last_event[device_id].get('time')}")
                else:
                    logger.error(f"Invalid document in list for device {device_id} events")
                    device_states.last_event[device_id] = None
            else:
                device_states.last_event[device_id] = None
                logger.debug(f"No events found for device {device_id} (empty list).")
        elif hasattr(col_snapshot, 'docs'):
            if col_snapshot.docs:
                last_doc = col_snapshot.docs[0]
                device_states.last_event[device_id] = {"id": last_doc.id, **last_doc.to_dict()}
                logger.debug(f"Updated last_event for {device_id}: Type={device_states.last_event[device_id].get('type')} Time={device_states.last_event[device_id].get('time')}")
            else:
                device_states.last_event[device_id] = None
                logger.debug(f"No events found for device {device_id}.")
        else:
            # Handle unexpected data types
            logger.error(f"Unexpected snapshot type for device {device_id} events: {type(col_snapshot)}")
            device_states.last_event[device_id] = None
            
    except Exception as e:
        logger.error(f"Error in _on_events_snapshot for {device_id}: {str(e)} - Snapshot type: {type(col_snapshot)}")
        device_states.last_event[device_id] = None

def _on_notifications_snapshot(col_snapshot, changes, read_time, device_id: str):
    """Callback function for notifications collection snapshot listener."""
    try:
        # Handle case where col_snapshot might be a list or other unexpected types
        if isinstance(col_snapshot, list):
            if len(col_snapshot) > 0:
                # If it's a list of documents, get the first one
                last_doc = col_snapshot[0]
                if hasattr(last_doc, 'id') and hasattr(last_doc, 'to_dict'):
                    device_states.last_notification[device_id] = {"id": last_doc.id, **last_doc.to_dict()}
                    logger.debug(f"Updated last_notification for {device_id}: Type={device_states.last_notification[device_id].get('type')} Time={device_states.last_notification[device_id].get('time')}")
                else:
                    logger.error(f"Invalid document in list for device {device_id} notifications")
                    device_states.last_notification[device_id] = None
            else:
                device_states.last_notification[device_id] = None
                logger.debug(f"No notifications found for device {device_id} (empty list).")
        elif hasattr(col_snapshot, 'docs'):
            if col_snapshot.docs:
                last_doc = col_snapshot.docs[0]
                device_states.last_notification[device_id] = {"id": last_doc.id, **last_doc.to_dict()}
                logger.debug(f"Updated last_notification for {device_id}: Type={device_states.last_notification[device_id].get('type')} Time={device_states.last_notification[device_id].get('time')}")
            else:
                device_states.last_notification[device_id] = None
                logger.debug(f"No notifications found for device {device_id}.")
        else:
            # Handle unexpected data types
            logger.error(f"Unexpected snapshot type for device {device_id} notifications: {type(col_snapshot)}")
            device_states.last_notification[device_id] = None
            
    except Exception as e:
        logger.error(f"Error in _on_notifications_snapshot for {device_id}: {str(e)} - Snapshot type: {type(col_snapshot)}")
        device_states.last_notification[device_id] = None

### Hàm quản lý Listener

def setup_device_listeners(device_id: str):
    """Sets up real-time listeners for a specific device's settings, last event, and last notification."""
    if not initialize_firebase() or not firestore_client:
        logger.error("Failed to initialize Firebase for listener setup.")
        return False

    # Prevent setting up duplicate listeners
    if f'{device_id}_settings' in device_states.listeners and \
       f'{device_id}_events' in device_states.listeners and \
       f'{device_id}_notifications' in device_states.listeners: # NEW check for notifications listener
        logger.debug(f"Listeners for device {device_id} already set up.")
        return True

    try:
        device_ref = firestore_client.collection('devices').document(device_id)
        
        # Listener for cryingThreshold
        def settings_callback(doc_snapshot, changes, read_time):
            _on_device_settings_snapshot(doc_snapshot, changes, read_time, device_id)
        
        device_listener_fn = device_ref.on_snapshot(settings_callback)
        device_states.listeners[f'{device_id}_settings'] = device_listener_fn
        logger.info(f"Listener for device {device_id} settings started.")

        # Listener for last_event (type crying or nocrying)
        events_query = device_ref.collection('events') \
                                 .where(filter=firestore.FieldFilter("type", "in", [EVENT_TYPES['CRYING'], EVENT_TYPES['NO_CRYING']])) \
                                 .order_by('time', direction=firestore.Query.DESCENDING) \
                                 .limit(1)

        def events_callback(col_snapshot, changes, read_time):
            _on_events_snapshot(col_snapshot, changes, read_time, device_id)

        events_listener_fn = events_query.on_snapshot(events_callback)
        device_states.listeners[f'{device_id}_events'] = events_listener_fn
        logger.info(f"Listener for device {device_id} events started.")

        # NEW: Listener for last_notification (only crying type)
        notifications_query = device_ref.collection('notifications') \
                                        .where(filter=firestore.FieldFilter("type", "==", EVENT_TYPES['CRYING'])) \
                                        .order_by('time', direction=firestore.Query.DESCENDING) \
                                        .limit(1)

        def notifications_callback(col_snapshot, changes, read_time):
            _on_notifications_snapshot(col_snapshot, changes, read_time, device_id)

        notifications_listener_fn = notifications_query.on_snapshot(notifications_callback)
        device_states.listeners[f'{device_id}_notifications'] = notifications_listener_fn
        logger.info(f"Listener for device {device_id} notifications started.")

        return True
    except Exception as e:
        logger.error(f"Error setting up listeners for device {device_id}: {str(e)}")
        return False

def stop_device_listeners(device_id: str):
    """Stops the real-time listeners for a specific device."""
    if f'{device_id}_settings' in device_states.listeners:
        device_states.listeners.pop(f'{device_id}_settings')()
        logger.info(f"Stopped settings listener for device {device_id}.")
    if f'{device_id}_events' in device_states.listeners:
        device_states.listeners.pop(f'{device_id}_events')()
        logger.info(f"Stopped events listener for device {device_id}.")
    if f'{device_id}_notifications' in device_states.listeners: # NEW: Stop notifications listener
        device_states.listeners.pop(f'{device_id}_notifications')()
        logger.info(f"Stopped notifications listener for device {device_id}.")

def stop_all_listeners():
    """Stops all active Firebase listeners."""
    for key, unsubscribe_fn in list(device_states.listeners.items()):
        try:
            unsubscribe_fn()
            logger.info(f"Stopped listener: {key}")
        except Exception as e:
            logger.error(f"Error stopping listener {key}: {str(e)}")
    device_states.listeners.clear()
    logger.info("All Firebase listeners stopped.")

## Hàm xử lý Logic (Đã được tối ưu hóa)

def create_notification_doc(timestamp: int, duration: float) -> Dict[str, Any]:
    """Create notification document data"""
    dt = convert_utc_timestamp_to_vn_datetime(timestamp)
    return {
        'type': EVENT_TYPES['CRYING'],
        'time': dt,
        'duration': duration,
        'imageUrl': ''
    }

def should_create_notification(last_notification: Optional[Dict[str, Any]],
                               current_timestamp: float,
                               cry_threshold: float,
                               duration: float) -> bool:
    """Determine if a new notification should be created"""
    if cry_threshold == 0:
        return False

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

        # --- OPTIMIZED: Get data from global state ---
        cry_threshold = device_states.crying_threshold.get(device_id, DEFAULT_CRY_THRESHOLD)
        last_event = device_states.last_event.get(device_id)
        last_notification = device_states.last_notification.get(device_id) # NEW: Get last_notification from state

        # Ensure listeners are set up for this device if not already (important for new devices)
        # This is crucial for lazy loading listeners for active devices.
        if device_id not in device_states.crying_threshold or \
           device_id not in device_states.last_event or \
           device_id not in device_states.last_notification: # NEW check
            logger.warning(f"Device {device_id} data not yet in global state. Attempting to set up listeners...")
            setup_device_listeners(device_id)
            # In a real async system, you might want a short delay or a mechanism
            # to ensure initial data is loaded before proceeding,
            # or a fallback to one-time fetch if immediate data is needed.
            # For this example, we proceed assuming eventual consistency.

        # Validate device exists in Firestore (still a one-time check)
        device_ref = firestore_client.collection('devices').document(device_id)
        if not device_ref.get().exists:
            logger.error(f"Device {device_id} does not exist in Firestore")
            return False

        # Get Firestore references (still needed for write operations)
        events_ref = device_ref.collection('events')
        notifications_ref = device_ref.collection('notifications')

        # Create new crying event if last event was NO_CRYING or no event exists
        if not last_event or last_event.get('type') == EVENT_TYPES['NO_CRYING']:
            new_event = {
                'type': EVENT_TYPES['CRYING'],
                'time': convert_utc_timestamp_to_vn_datetime(current_timestamp)
            }
            events_ref.add(new_event) # This write will trigger the event listener
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
            
            # Use last_notification from global state
            # If should_create_notification needs a new notification, add it
            if should_create_notification(last_notification, current_timestamp, cry_threshold, duration):
                notification_doc_data = {
                    'type': EVENT_TYPES['CRYING'],
                    'time': convert_utc_timestamp_to_vn_datetime(current_timestamp),
                    'duration': duration,
                    'imageUrl': ''
                }
                notifications_ref.add(notification_doc_data) # This write will trigger the notification listener

                tokens_with_info, deviceId = await get_fcm_tokens_for_device(device_id)
                if tokens_with_info:
                    await send_crying_notification_fcm(deviceId, tokens_with_info, duration)
                else:
                    logger.warning(f"No FCM tokens found for device {device_id}")
                logger.info(f"Created Crying notification with duration {duration:.2f}s for device {device_id}")
            elif cry_threshold == 0:
                logger.info(f"Cry threshold is 0 for device {device_id}, notifications disabled but events still tracked")

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
        current_timestamp = float(data['timestamp'])

        # --- OPTIMIZED: Get last event from global state ---
        last_event = device_states.last_event.get(device_id)

        # Ensure listeners are set up for this device if not already
        if device_id not in device_states.last_event:
            logger.warning(f"Device {device_id} event data not yet in global state. Attempting to set up listeners...")
            setup_device_listeners(device_id)

        events_ref = firestore_client.collection('devices').document(device_id).collection('events')

        # Add NoCrying event regardless of last event type if we have a valid lastCryTimestamp
        if data.get('lastCryTimestamp') is not None:
            should_add_event = True
            if last_event and last_event.get('type') == EVENT_TYPES['NO_CRYING']:
                last_event_timestamp = last_event.get('time').timestamp() if last_event.get('time') else 0
                if abs(current_timestamp - last_event_timestamp) < 2:
                    should_add_event = False
                    logger.debug(f"Skipping duplicate no-cry event for device {device_id}")

            if should_add_event:
                events_ref.add({
                    'type': EVENT_TYPES['NO_CRYING'],
                    'time': convert_utc_timestamp_to_vn_datetime(current_timestamp)
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
                                     .where(filter=FieldFilter('deviceId', '==', device_id)) \
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

# Example of how you might initialize listeners for active devices
# This would typically be called once when your backend service starts.
async def start_service_and_listeners():
    if not initialize_firebase():
        logger.error("Service cannot start without Firebase initialization.")
        return

    try:
        # Fetch initial list of devices to set up listeners
        # In a production system, you might have a more robust way to manage active devices
        # (e.g., listening to a 'status' field in device docs, or a separate 'active_devices' collection)
        all_devices = firestore_client.collection('devices').stream()
        for doc in all_devices:
            device_id = doc.id
            setup_device_listeners(device_id)
        logger.info("Finished setting up initial listeners for all devices.")
    except Exception as e:
        logger.error(f"Error fetching initial devices to set up listeners: {str(e)}")

# Remember to call stop_all_listeners() when your application shuts down gracefully
# For example, in a FastAPI/Flask app shutdown hook, or a SIGTERM handler.