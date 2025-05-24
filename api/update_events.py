#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to update Firestore events collection based on specific time conditions:
1. Events with time in the future: move back by 1 day
2. Events on May 14-15, 2025: move forward by 7 days
"""

import firebase_admin
from firebase_admin import credentials, firestore
import os
import logging
from datetime import datetime, timedelta
import pytz

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Firebase
def initialize_firebase() -> bool:
    """Initialize Firebase connection if not already initialized"""
    try:
        cred_path = os.path.join(os.path.dirname(__file__), "firebase-credentials.json")
        if not os.path.exists(cred_path):
            logger.error(f"Firebase credentials file not found at: {cred_path}")
            return False
            
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred)
        return True
        
    except Exception as e:
        logger.error(f"Error initializing Firebase: {str(e)}")
        return False

def update_events():
    """Update events based on time conditions"""
    if not initialize_firebase():
        logger.error("Failed to initialize Firebase")
        return

    # Get Firestore client
    db = firestore.client()
    
    # Get current time in Vietnam timezone
    vn_tz = pytz.timezone('Asia/Ho_Chi_Minh')
    current_time = datetime.now(vn_tz)
    
    # Define target dates for May 14-15, 2025
    may_14 = datetime(2025, 5, 14, tzinfo=vn_tz)
    may_15 = datetime(2025, 5, 15, tzinfo=vn_tz)
    may_15_end = datetime(2025, 5, 15, 23, 59, 59, tzinfo=vn_tz)

    # Get all devices
    devices = db.collection('devices').stream()
    
    update_count = 0
    for device in devices:
        device_id = device.id
        logger.info(f"Processing device: {device_id}")
        
        # Get all events for the device
        events_ref = db.collection('devices').document(device_id).collection('events')
        events = events_ref.stream()
        
        for event in events:
            event_data = event.to_dict()
            event_time = event_data.get('time')
            if not event_time:
                continue
            
            if event_time == firestore.SERVER_TIMESTAMP:
                continue

            # Convert to datetime if it's a Timestamp
            if isinstance(event_time, datetime):
                event_datetime = event_time
            else:
                event_datetime = event_time.astimezone(vn_tz)

            new_time = None
            
            # Check if event is in the future
            if event_datetime > current_time:
                new_time = event_datetime - timedelta(days=1)
                logger.info(f"Moving future event back 1 day: {event_datetime} -> {new_time}")
            
            # Check if event is on May 14-15, 2025
            elif may_14 <= event_datetime <= may_15_end:
                new_time = event_datetime + timedelta(days=7)
                logger.info(f"Moving May 14-15 event forward 7 days: {event_datetime} -> {new_time}")
            
            # Update the event if needed
            if new_time:
                event.reference.update({'time': new_time})
                update_count += 1
    
    logger.info(f"Updated {update_count} events")

if __name__ == "__main__":
    update_events()
