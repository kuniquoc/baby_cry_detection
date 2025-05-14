import logging
from typing import Dict, Any
import asyncio
import threading
import time

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from websocket.connection_manager import ConnectionManager
from firebase_service import send_cry_notification, send_nocry_notification

logger = logging.getLogger(__name__)

class CryDetectionService:
    def __init__(self, connection_manager: ConnectionManager, no_cry_check_seconds: int = 10):
        self.manager = connection_manager
        self.no_cry_check_seconds = no_cry_check_seconds
        self.cry_status_tracker: Dict[str, Dict[str, Any]] = {}

    def init_device_tracking(self, client_id: str):
        """Initialize or reset cry status tracking for a device"""
        self.cry_status_tracker[client_id] = {
            'last_cry_time': None,
            'checking_no_cry': False,
            'no_cry_timer': None
        }

    def cleanup_device_tracking(self, client_id: str):
        """Clean up tracking resources for a device"""
        if client_id in self.cry_status_tracker:
            if self.cry_status_tracker[client_id]['no_cry_timer'] is not None:
                try:
                    self.cry_status_tracker[client_id]['no_cry_timer'].cancel()
                except:
                    pass
            del self.cry_status_tracker[client_id]

    async def process_cry_detection(self, client_id: str, timestamp: float, confidence: float) -> bool:
        """Process cry detection event and handle notifications"""
        device_status = self.cry_status_tracker[client_id]
        device_status['last_cry_time'] = timestamp

        # Cancel any existing no-cry timer
        if device_status['checking_no_cry'] and device_status['no_cry_timer'] is not None:
            if device_status['no_cry_timer'].is_alive():
                device_status['no_cry_timer'].cancel()
            device_status['checking_no_cry'] = False
            device_status['no_cry_timer'] = None
            logger.info(f"Cancelled no-cry timer for device {client_id} - new crying detected")

        # Check Firebase events and send notification if needed
        notification_sent = await send_cry_notification({
            'timestamp': timestamp,
            'deviceId': client_id,
            'confidence': confidence
        })

        if notification_sent:
            await self.manager.send_message(client_id, {
                "type": "alert",
                "timestamp": timestamp,
                "message": "Crying detected!",
                "confidence": confidence,
                "deviceId": client_id
            })
            logger.info(f"Cry event created at timestamp {timestamp} for device {client_id}")

        # Start no-cry timer if not already checking
        if device_status['last_cry_time'] is not None and not device_status['checking_no_cry']:
            device_status['checking_no_cry'] = True
            no_cry_timer = threading.Timer(
                self.no_cry_check_seconds,
                self._schedule_no_cry_check,
                args=[client_id, timestamp]
            )
            no_cry_timer.daemon = True
            no_cry_timer.start()
            device_status['no_cry_timer'] = no_cry_timer
            logger.info(f"Started no-cry timer for device {client_id}")

        return notification_sent

    def _schedule_no_cry_check(self, client_id: str, last_cry_timestamp: float):
        """Schedule the no-cry check to run in the event loop"""
        loop = asyncio.get_event_loop()
        loop.create_task(self._check_for_no_cry(client_id, last_cry_timestamp))

    async def _check_for_no_cry(self, client_id: str, last_cry_timestamp: float):
        """Check if crying has stopped after the timeout period"""
        try:
            if client_id not in self.cry_status_tracker:
                logger.warning(f"Device {client_id} not in tracker when checking for no-cry")
                return

            device_status = self.cry_status_tracker[client_id]

            # If this is an outdated timer (a newer cry was detected), do nothing
            if device_status['last_cry_time'] != last_cry_timestamp:
                logger.info(f"No-cry check skipped for device {client_id} - newer cry detected")
                return

            # If we get here, the timeout period has passed with no new cry detection
            current_time = time.time()

            # Send no-cry notification
            notification_sent = await send_nocry_notification({
                'timestamp': current_time,
                'deviceId': client_id,
                'lastCryTimestamp': last_cry_timestamp
            })

            if notification_sent:
                logger.info(
                    f"No-cry event detected at {current_time} for device {client_id}. "
                    f"Last cry was at {last_cry_timestamp}, "
                    f"{current_time - last_cry_timestamp:.2f}s ago"
                )

            # Reset the checking flag
            device_status['checking_no_cry'] = False
            device_status['no_cry_timer'] = None

        except Exception as e:
            logger.error(f"Error in check_for_no_cry: {str(e)}")