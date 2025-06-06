import logging
from typing import Dict, Any
import asyncio
import time

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from websocket.connection_manager import ConnectionManager
from firebase_service import send_cry_notification, send_nocry_notification

from utils.date_utils import convert_utc_timestamp_to_vn_datetime

logger = logging.getLogger(__name__)

class CryDetectionService:
    def __init__(self, connection_manager: ConnectionManager, no_cry_check_seconds: int = 10):
        self.manager = connection_manager
        self.no_cry_check_seconds = no_cry_check_seconds
        self.cry_status_tracker: Dict[str, Dict[str, Any]] = {}
        self.check_task = None

    async def start_periodic_check(self):
        """Start periodic checking for no-cry events"""
        self.check_task = asyncio.create_task(self._periodic_check())

    async def stop_periodic_check(self):
        """Stop the periodic checking task"""
        if self.check_task:
            self.check_task.cancel()
            try:
                await self.check_task
            except asyncio.CancelledError:
                pass
            self.check_task = None

    async def _periodic_check(self):
        """Check all devices every second for no-cry events"""
        logger.info("Periodic check for no-cry events started")
        while True:
            try:
                current_time = time.time()  # Use Unix timestamp instead of loop time
                tracked_devices = list(self.cry_status_tracker.items())
                
                if tracked_devices:
                    logger.debug(f"Checking {len(tracked_devices)} tracked devices for no-cry events")
                
                for client_id, device_status in tracked_devices:
                    if device_status['last_cry_time'] is not None and device_status.get('cry_confirmed', False):
                        time_since_last_cry = current_time - device_status['last_cry_time']
                        logger.debug(f"Device {client_id}: {time_since_last_cry:.1f}s since last cry, no_cry_sent: {device_status.get('no_cry_sent', False)}")
                        
                        if time_since_last_cry >= self.no_cry_check_seconds and not device_status.get('no_cry_sent', False):
                            logger.info(f"Triggering no-cry event for device {client_id} after {time_since_last_cry:.1f}s")
                            # Send no-cry notification
                            notification_sent = await send_nocry_notification({
                                'timestamp': current_time,
                                'deviceId': client_id,
                                'lastCryTimestamp': device_status['last_cry_time']
                            })

                            if notification_sent:
                                try:
                                    await self.manager.send_message(client_id, {
                                        "type": "alert",
                                        "timestamp": current_time,
                                        "message": "No crying detected",
                                        "deviceId": client_id
                                    })
                                except Exception as e:
                                    # Ignore WebSocket errors (connection might be closed)
                                    logger.debug(f"Could not send no-cry message to {client_id}: {str(e)}")
                                    
                                logger.info(
                                    f"No-cry event detected at {convert_utc_timestamp_to_vn_datetime(current_time)} for device {client_id}. "
                                    f"Last cry was at {device_status['last_cry_time']}, "
                                    f"{time_since_last_cry:.2f}s ago"
                                )
                                device_status['no_cry_sent'] = True
                            else:
                                logger.warning(f"Failed to send no-cry notification for device {client_id}")

            except Exception as e:
                logger.error(f"Error in periodic check: {str(e)}")
            
            await asyncio.sleep(1)  # Wait for 1 second before next check

    def init_device_tracking(self, client_id: str):
        """Initialize or reset cry status tracking for a device"""
        self.cry_status_tracker[client_id] = {
            'last_cry_time': None,
            'no_cry_sent': False,
            'consecutive_cry_timestamps': [],  # Store timestamps of consecutive cry detections
            'cry_confirmed': False
        }

    def cleanup_device_tracking(self, client_id: str):
        """Clean up tracking resources for a device"""
        if client_id in self.cry_status_tracker:
            del self.cry_status_tracker[client_id]

    async def process_cry_detection(self, client_id: str, timestamp: float, confidence: float) -> bool:
        """Process cry detection event and handle notifications"""
        if client_id not in self.cry_status_tracker:
            self.init_device_tracking(client_id)

        device_status = self.cry_status_tracker[client_id]
        
        # Clean up old timestamps first
        self.cleanup_old_timestamps(client_id, timestamp)
        
        # Add current timestamp to consecutive cry list
        device_status['consecutive_cry_timestamps'].append(timestamp)
        
        # Keep only the last 10 timestamps to avoid memory issues
        if len(device_status['consecutive_cry_timestamps']) > 10:
            device_status['consecutive_cry_timestamps'] = device_status['consecutive_cry_timestamps'][-10:]
        
        # Check if we have at least 3 consecutive cry detections
        consecutive_count = self._count_consecutive_cries(device_status['consecutive_cry_timestamps'])
        
        logger.debug(f"Cry detected for device {client_id} (consecutive: {consecutive_count}/3) at {convert_utc_timestamp_to_vn_datetime(timestamp)} with confidence {confidence:.2f}")
        
        # Check if we have 3 consecutive cry detections and not already confirmed
        if consecutive_count >= 3 and not device_status['cry_confirmed']:
            # Update last cry time and reset no_cry_sent flag
            device_status['last_cry_time'] = timestamp
            device_status['no_cry_sent'] = False
            device_status['cry_confirmed'] = True
            
            logger.info(f"Cry CONFIRMED for device {client_id} after 3 consecutive detections at {convert_utc_timestamp_to_vn_datetime(timestamp)} with confidence {confidence:.2f}")

            # Send cry notification to Firebase
            notification_sent = await send_cry_notification({
                'timestamp': timestamp,
                'deviceId': client_id,
                'confidence': confidence
            })

            return notification_sent
        else:
            logger.debug(f"Cry detected but not confirmed yet for device {client_id} (need {3 - consecutive_count} more consecutive detections)")
            return False

    def reset_cry_detection(self, client_id: str):
        """Reset cry detection status when no cry is detected"""
        if client_id in self.cry_status_tracker:
            device_status = self.cry_status_tracker[client_id]
            if len(device_status['consecutive_cry_timestamps']) > 0:
                logger.debug(f"Resetting cry timestamps for device {client_id} (had {len(device_status['consecutive_cry_timestamps'])} timestamps)")
            device_status['consecutive_cry_timestamps'] = []
            device_status['cry_confirmed'] = False

    async def send_final_no_cry_events(self):
        """Send no-cry events for all tracked devices before shutdown (Firebase only)"""
        logger.info(f"Sending final no-cry events for {len(self.cry_status_tracker)} tracked devices")
        current_time = time.time()  # Use Unix timestamp instead of loop time
        for client_id, device_status in list(self.cry_status_tracker.items()):
            if device_status['last_cry_time'] is not None and device_status.get('cry_confirmed', False) and not device_status.get('no_cry_sent', False):
                logger.info(f"Sending final no-cry event for device {client_id}")
                # Send no-cry notification for this device (Firebase only, no WebSocket)
                notification_sent = await send_nocry_notification({
                    'timestamp': current_time,
                    'deviceId': client_id,
                    'lastCryTimestamp': device_status['last_cry_time']
                })

                if notification_sent:
                    # Don't send WebSocket messages during shutdown as connections are already closed
                    logger.info(
                        f"Final no-cry event sent to Firebase for device {client_id} during shutdown. "
                        f"Last cry was at {device_status['last_cry_time']}, "
                        f"{current_time - device_status['last_cry_time']:.2f}s ago"
                    )
                else:
                    logger.warning(f"Failed to send final no-cry event for device {client_id}")
            else:
                if device_status['last_cry_time'] is None:
                    logger.debug(f"Device {client_id} has no cry recorded, skipping final no-cry event")
                else:
                    logger.debug(f"Device {client_id} already has no-cry event sent, skipping")

    async def handle_client_disconnect(self, client_id: str, skip_websocket: bool = False):
        """Handle client disconnection, sending final no-cry event if needed"""
        if client_id in self.cry_status_tracker:
            device_status = self.cry_status_tracker[client_id]
            if device_status['last_cry_time'] is not None and device_status.get('cry_confirmed', False) and not device_status.get('no_cry_sent', False):
                current_time = time.time()  # Use Unix timestamp instead of loop time

                logger.info(f"Create final no_cry event when client disconnect for device {client_id}")
                # Send no-cry notification for this device
                notification_sent = await send_nocry_notification({
                    'timestamp': current_time,
                    'deviceId': client_id,
                    'lastCryTimestamp': device_status['last_cry_time']
                })

                if notification_sent:
                    if not skip_websocket:
                        try:
                            await self.manager.send_message(client_id, {
                                "type": "alert",
                                "timestamp": current_time,
                                "message": "No crying detected (client disconnected)",
                                "deviceId": client_id
                            })
                        except Exception as e:
                            # Ignore WebSocket errors during disconnect
                            logger.debug(f"Could not send disconnect message: {str(e)}")
                    logger.info(
                        f"No-cry event sent for device {client_id} during disconnect. "
                        f"Last cry was at {device_status['last_cry_time']}, "
                        f"{current_time - device_status['last_cry_time']:.2f}s ago"
                    )
                else:
                    logger.warning(f"Failed to send no-cry notification for device {client_id} during disconnect")
            else:
                if device_status['last_cry_time'] is None:
                    logger.debug(f"No cry detected for device {client_id}, skipping no-cry event")
                else:
                    logger.debug(f"No-cry event already sent for device {client_id}, skipping")

    def _count_consecutive_cries(self, timestamps: list, max_gap: float = 3.0) -> int:
        """Count consecutive cry detections based on timestamp gaps
        
        Args:
            timestamps: List of cry detection timestamps
            max_gap: Maximum allowed gap between consecutive cries (seconds)
            
        Returns:
            Number of consecutive cry detections from the end of the list
        """
        if len(timestamps) < 2:
            return len(timestamps)
        
        # Sort timestamps to ensure proper order
        sorted_timestamps = sorted(timestamps)
        
        # Count consecutive detections from the end
        consecutive_count = 1  # Start with the last detection
        
        for i in range(len(sorted_timestamps) - 2, -1, -1):
            current_time = sorted_timestamps[i]
            next_time = sorted_timestamps[i + 1]
            
            # Check if the gap is within the allowed range
            if next_time - current_time <= max_gap:
                consecutive_count += 1
            else:
                # Gap is too large, break the consecutive chain
                break
                
        return consecutive_count

    def cleanup_old_timestamps(self, client_id: str, current_timestamp: float, max_age: float = 10.0):
        """Clean up old timestamps that are too old to be relevant
        
        Args:
            client_id: Device client ID
            current_timestamp: Current timestamp
            max_age: Maximum age of timestamps to keep (seconds)
        """
        if client_id in self.cry_status_tracker:
            device_status = self.cry_status_tracker[client_id]
            old_count = len(device_status['consecutive_cry_timestamps'])
            
            # Filter out timestamps older than max_age
            device_status['consecutive_cry_timestamps'] = [
                ts for ts in device_status['consecutive_cry_timestamps'] 
                if current_timestamp - ts <= max_age
            ]
            
            new_count = len(device_status['consecutive_cry_timestamps'])
            if old_count > new_count:
                logger.debug(f"Cleaned up {old_count - new_count} old timestamps for device {client_id}")

    def check_and_reset_if_gap_too_large(self, client_id: str, current_timestamp: float, max_gap: float = 5.0):
        """Check if there's a large gap since last cry and reset if needed
        
        Args:
            client_id: Device client ID
            current_timestamp: Current timestamp
            max_gap: Maximum allowed gap before reset (seconds)
        """
        if client_id in self.cry_status_tracker:
            device_status = self.cry_status_tracker[client_id]
            timestamps = device_status['consecutive_cry_timestamps']
            
            if len(timestamps) > 0:
                last_timestamp = max(timestamps)
                gap = current_timestamp - last_timestamp
                
                if gap > max_gap:
                    logger.debug(f"Large gap detected ({gap:.1f}s) for device {client_id}, resetting cry detection")
                    self.reset_cry_detection(client_id)