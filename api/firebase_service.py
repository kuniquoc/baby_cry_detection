import firebase_admin
from firebase_admin import credentials, firestore
import os
import logging
from fcm_service import send_crying_notification_fcm


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Firebase credentials and initialization
firebase_initialized = False
firestore_client = None
snapshot_callbacks = {}

def initialize_firebase():
    """Initialize Firebase connection"""
    global firebase_initialized, firestore_client
    
    if firebase_initialized:
        return True
    
    try:
        # Look for Firebase credentials file in the api directory
        cred_path = os.path.join(os.path.dirname(__file__), "firebase-credentials.json")
        
        if not os.path.exists(cred_path):
            logger.error(f"Firebase credentials file not found at: {cred_path}")
            return False
            
        # Initialize Firebase app
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred)
        
        # Initialize Firestore client
        firestore_client = firestore.client()
        
        firebase_initialized = True
        logger.info("Firebase successfully initialized")
        return True
        
    except Exception as e:
        logger.error(f"Error initializing Firebase: {str(e)}")
        return False

async def send_cry_notification(notification_data):
    """
    Send a cry detection notification to Firebase
    
    Parameters:
    ----------
    notification_data : dict
        Dictionary containing:
        - timestamp: float (seconds)
        - deviceId: string
        - message: str (optional)
        - audioFile: str (optional)
        - confidence: float (optional)
    
    Returns:
    -------
    bool
        True if notification was successfully sent, False otherwise
    """
    if not notification_data:
        logger.error("Notification data is None or empty")
        return False
        
    if not initialize_firebase():
        logger.error("Failed to initialize Firebase")
        return False
    
    try:
        # Get deviceId from notification data
        device_id = notification_data.get('deviceId')
        current_timestamp = notification_data.get('timestamp')
        
        if not device_id or not current_timestamp:
            logger.error("DeviceId and timestamp are required but not provided in notification data")
            return False
            
        # Convert timestamp from seconds to milliseconds for Firestore
        current_timestamp_ms = int(current_timestamp * 1000)
        
        # Create Firestore client
        global firestore_client
        
        if not firestore_client:
            logger.error("Firestore client is not initialized")
            return False
            
        # References to device collections
        device_ref = firestore_client.collection('devices').document(device_id)
        events_ref = device_ref.collection('events')
        notifications_ref = device_ref.collection('notifications')
        
        # Get device settings
        try:
            device_doc = device_ref.get()
            if device_doc and device_doc.exists:
                device_data = device_doc.to_dict()
                # Get crying threshold from device settings (default: 10 seconds)
                cry_threshold = device_data.get('cryingThreshold', 10) if device_data else 10
            else:
                cry_threshold = 10  # Default value if device document doesn't exist
                logger.warning(f"Device {device_id} not found, using default crying threshold of {cry_threshold}s")
        except Exception as e:
            cry_threshold = 10  # Default value if there's an error
            logger.error(f"Error retrieving device settings: {str(e)}, using default crying threshold of {cry_threshold}s")
        
        # Get the latest events with type Crying or NoCrying
        # Update: Using positional arguments for filtering
        latest_events = events_ref.where("type", "in", ['Crying', 'NoCrying']) \
                               .order_by('time', direction=firestore.Query.DESCENDING) \
                               .limit(1) \
                               .stream()
        
        last_event = None
        for event in latest_events:
            if event:
                last_event = event.to_dict()
                last_event['id'] = event.id
                break
        
        # Define crying event data with millisecond timestamp
        event_data = {
            'type': 'Crying',
            'time': current_timestamp_ms
        }
        
        # If the last event was NoCrying or there's no previous event, add a new Crying event and return
        if not last_event or last_event.get('type') == 'NoCrying':
            # Add the new crying event
            events_ref.add(event_data)
            logger.info(f"Added new Crying event at {current_timestamp} ({current_timestamp_ms} ms) for device {device_id}")
            return True  # Exit function - no notification needed when adding a new Crying event
        
        # If we reach here, the last event was Crying, so we just process notification logic
        # without adding a new event
        
        # Convert last crying time from milliseconds back to seconds for time difference calculation
        last_crying_time_ms = last_event.get('time')
        if last_crying_time_ms is None:
            logger.warning(f"Last event for device {device_id} has no 'time' field, using current timestamp")
            last_crying_time_ms = current_timestamp_ms
            
        # Convert from ms back to seconds for calculation
        last_crying_time = last_crying_time_ms / 1000
        time_difference = current_timestamp - last_crying_time
        
        # Check for recent notifications after the last crying event
        recent_notifications = notifications_ref.where("type", "==", "Crying") \
                                             .where("time", ">", last_crying_time_ms) \
                                             .order_by('time', direction=firestore.Query.DESCENDING) \
                                             .limit(1) \
                                             .stream()
        
        last_notification = None
        for notif in recent_notifications:
            if notif:
                last_notification = notif.to_dict()
                last_notification['id'] = notif.id
                break
        
        # No recent notifications found after the last crying event, create a new one
        if not last_notification:
            # Calculate time difference since last crying event
            time_difference = current_timestamp - last_crying_time
            
            # Only create notification if crying duration exceeds threshold
            if time_difference >= cry_threshold:
                # Create new notification with millisecond timestamp
                notification_doc_data = {
                    'type': 'Crying',
                    'time': current_timestamp_ms,
                    'duration': time_difference,  # Duration still in seconds
                    'imageUrl': ''  # Empty as per schema requirement
                }
                
                # Add notification to Firestore
                notification_ref = notifications_ref.add(notification_doc_data)
                logger.info(f"Created new Crying notification with duration {time_difference:.2f}s time diff: {time_difference:.2f}s >= threshold: {cry_threshold}s for device {device_id}")
                
                tokens_with_info, deviceId = await get_fcm_tokens_for_device(device_id)
                if tokens_with_info:
                    # Send notification to all tokens
                    await send_crying_notification_fcm(deviceId, tokens_with_info, time_difference)
                else:
                    logger.warning(f"No FCM tokens found for device {device_id}, skipping notification")
                
            else:
                logger.info(f"Skipping notification creation: time difference {time_difference:.2f}s is less than threshold {cry_threshold}s for device {device_id}")
            return True
            
        # Recent notification found, check if the time difference exceeds the threshold
        if time_difference >= cry_threshold:
            last_notification_time_ms = last_notification.get('time')
            if last_notification_time_ms is None:
                logger.warning(f"Last notification for device {device_id} has no 'time' field, using current timestamp")
                last_notification_time_ms = current_timestamp_ms
            
            # Convert notification time from ms to seconds for calculation    
            last_notification_time = last_notification_time_ms / 1000
            notification_time_diff = current_timestamp - last_notification_time
            
            if notification_time_diff < cry_threshold:
                logger.info(f"Skipping notification creation: recent notification exists within threshold ({notification_time_diff:.2f}s < {cry_threshold}s)")
                return True
                
            # Calculate duration in seconds
            duration = current_timestamp - last_crying_time
            
            # Create new notification with millisecond timestamp
            notification_doc_data = {
                'type': 'Crying',
                'time': current_timestamp_ms,
                'duration': duration,  # Duration still in seconds
                'imageUrl': ''  # Empty as per schema requirement
            }
            
            # Add notification to Firestore
            notifications_ref.add(notification_doc_data)
            
            # Send push notification via FCM
            tokens_with_info, deviceId = await get_fcm_tokens_for_device(device_id)
            if tokens_with_info:
                # Send notification to all tokens
                await send_crying_notification_fcm(deviceId, tokens_with_info, duration)
            else:
                logger.warning(f"No FCM tokens found for device {device_id}, skipping notification")
            
            logger.info(f"Created new Crying notification with duration {duration:.2f}s for device {device_id}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error processing notification: {str(e)}")
        return False

async def send_nocry_notification(data):
    """
    Update Firebase when baby has stopped crying (only adds an event, no notification)
    
    Parameters:
    ----------
    data : dict
        Data to send to Firebase including:
        - timestamp: float - When the no-cry event was detected
        - deviceId: str - Device ID for Firebase
        - lastCryTimestamp: float - When the last cry was detected
    
    Returns:
    -------
    bool
        True if event was added successfully, False otherwise
    """
    if not data:
        logger.error("No-cry data is None or empty")
        return False
        
    if not initialize_firebase():
        logger.error("Failed to initialize Firebase")
        return False
        
    try:
        device_id = data.get('deviceId')
        current_timestamp = data.get('timestamp')
        last_cry_timestamp = data.get('lastCryTimestamp')
        
        if not all([device_id, current_timestamp, last_cry_timestamp]):
            logger.error("DeviceId, timestamp and lastCryTimestamp are required for no-cry event")
            return False
            
        # Convert timestamp from seconds to milliseconds for Firestore
        current_timestamp_ms = int(current_timestamp * 1000)
        
        # Get references to Firestore collections
        global firestore_client
        
        if not firestore_client:
            logger.error("Firestore client is not initialized")
            return False
            
        device_ref = firestore_client.collection('devices').document(device_id)
        events_ref = device_ref.collection('events')
        
        # Get the most recent Crying/NoCrying event
        latest_events = events_ref.where("type", "in", ["Crying", "NoCrying"]) \
                              .order_by('time', direction=firestore.Query.DESCENDING) \
                              .limit(1) \
                              .stream()
        
        last_event = None
        for event in latest_events:
            if event:
                last_event = event.to_dict()
                if last_event:
                    last_event['id'] = event.id
                break
            
        # Only add NoCrying event if the last event was Crying
        if last_event and last_event.get('type') == 'Crying':
            # Add NoCrying event with millisecond timestamp
            nocry_event = {
                'type': 'NoCrying',
                'time': current_timestamp_ms
            }
            
            events_ref.add(nocry_event)
            
            logger.info(f"Added NoCrying event at {current_timestamp} ({current_timestamp_ms} ms) for device {device_id}. "
                      f"Last cry was at {last_cry_timestamp}")
            return True
        else:
            event_type = last_event.get('type') if last_event else 'None'
            logger.info(f"Skipped adding NoCrying event for device {device_id} - "
                      f"last event was already {event_type}")
            return True
            
    except Exception as e:
        logger.error(f"Error sending no-cry event: {str(e)}")
        return False

def setup_device_listener(device_id, callback_function):
    """
    Set up a real-time listener for changes to a specific device's data
    
    Parameters:
    ----------
    device_id : str
        The ID of the device to listen for changes
    callback_function : function
        The function to call when changes are detected
        Function signature should be: callback(changes)
        
    Returns:
    -------
    bool
        True if listener was successfully set up, False otherwise
    """
    if not device_id:
        logger.error("Device ID is required for setting up a listener")
        return False
        
    if not callback_function:
        logger.error("Callback function is required for setting up a listener")
        return False
        
    if not initialize_firebase():
        logger.error("Failed to initialize Firebase")
        return False
    
    try:
        # Check if firestore_client was initialized properly
        global firestore_client
        if not firestore_client:
            logger.error("Firestore client is not initialized")
            return False
            
        # Create a reference to the device document
        device_ref = firestore_client.collection('devices').document(device_id)
        
        # Set up the snapshot listener
        def on_snapshot(doc_snapshot, changes, read_time):
            callback_function(doc_snapshot, changes, read_time)
        
        # Watch the document
        snapshot_callbacks[device_id] = device_ref.on_snapshot(on_snapshot)
        
        logger.info(f"Snapshot listener set up for device {device_id}")
        return True
        
    except Exception as e:
        logger.error(f"Error setting up device listener: {str(e)}")
        return False

def setup_events_listener(device_id, callback_function):
    """
    Set up a real-time listener for new events for a specific device
    
    Parameters:
    ----------
    device_id : str
        The ID of the device to listen for events
    callback_function : function
        The function to call when new events are added
        Function signature should be: callback(event_snapshot)
        
    Returns:
    -------
    bool
        True if listener was successfully set up, False otherwise
    """
    if not device_id:
        logger.error("Device ID is required for setting up an events listener")
        return False
        
    if not callback_function:
        logger.error("Callback function is required for setting up an events listener")
        return False
        
    if not initialize_firebase():
        logger.error("Failed to initialize Firebase")
        return False
    
    try:
        # Check if firestore_client was initialized properly
        global firestore_client
        if not firestore_client:
            logger.error("Firestore client is not initialized")
            return False
            
        # Create a reference to the events collection
        events_ref = firestore_client.collection('devices').document(device_id).collection('events')
        
        # Set up the snapshot listener
        def on_snapshot(coll_snapshot, changes, read_time):
            for change in changes:
                if change and change.type and change.type.name == 'ADDED' and change.document:
                    callback_function(change.document)
        
        # Watch the collection
        listener_key = f"{device_id}_events"
        snapshot_callbacks[listener_key] = events_ref.on_snapshot(on_snapshot)
        
        logger.info(f"Events listener set up for device {device_id}")
        return True
        
    except Exception as e:
        logger.error(f"Error setting up events listener: {str(e)}")
        return False

def stop_listener(device_id, collection_name=None):
    """
    Stop a listener for a specific device
    
    Parameters:
    ----------
    device_id : str
        The ID of the device
    collection_name : str, optional
        The name of the collection to stop listening to.
        If None, stops the device document listener.
        
    Returns:
    -------
    bool
        True if listener was successfully stopped, False otherwise
    """
    try:
        listener_key = f"{device_id}_{collection_name}" if collection_name else device_id
        
        if listener_key in snapshot_callbacks:
            # Call the unsubscribe function
            snapshot_callbacks[listener_key]()
            # Remove from callbacks dict
            del snapshot_callbacks[listener_key]
            logger.info(f"Listener stopped for {listener_key}")
            return True
        else:
            logger.warning(f"No listener found for {listener_key}")
            return False
            
    except Exception as e:
        logger.error(f"Error stopping listener: {str(e)}")
        return False

async def get_fcm_tokens_for_device(device_id):
    """
    Get all FCM tokens associated with a device through its connections
    
    Parameters:
    ----------
    device_id : str
        The ID of the device to get tokens for
        
    Returns:
    -------
    list
        List of fcm tokens associated with the device's connections
    """
    try:
        if not initialize_firebase():
            logger.error("Failed to initialize Firebase")
            return []
            
        global firestore_client
        if not firestore_client:
            logger.error("Firestore client is not initialized")
            return []
            
        fcm_tokens = []
        
        # Get all connections for the device
        connections = firestore_client.collection('connections') \
                                   .where('deviceId', '==', device_id) \
                                   .stream()
                                   
        # For each connection, get the user and their FCM tokens
        for connection in connections:
            conn_data = connection.to_dict()
            user_id = conn_data.get('userId')
            conn_name = conn_data.get('name', '')
            
            if user_id:
                # Get user document
                user_doc = firestore_client.collection('users').document(user_id).get()
                if user_doc.exists:
                    user_data = user_doc.to_dict()
                    fcm_tokens_data = user_data.get('fcmTokens', [])
                    # Add connection name to each token for use in notification
                    tokens_with_info = [(token, user_data.get('language', 'en'), conn_name) 
                                      for token in fcm_tokens_data]
                    fcm_tokens.extend(tokens_with_info)
        
        return fcm_tokens, device_id
        
    except Exception as e:
        logger.error(f"Error getting FCM tokens for device {device_id}: {str(e)}")
        return []