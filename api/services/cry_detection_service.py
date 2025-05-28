import logging
from typing import Dict, Any
import asyncio

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from websocket.connection_manager import ConnectionManager
from firebase_service import send_cry_notification, send_nocry_notification

from utils.date_utils import convert_utc_timestamp_to_vn_datetime

logger = logging.getLogger(__name__)

class CryDetectionService:
    def __init__(self, connection_manager: ConnectionManager, no_cry_check_seconds: int = 5):
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
        while True:
            try:
                current_time = asyncio.get_event_loop().time()
                for client_id, device_status in list(self.cry_status_tracker.items()):
                    if device_status['last_cry_time'] is not None:
                        time_since_last_cry = current_time - device_status['last_cry_time']
                        if time_since_last_cry >= self.no_cry_check_seconds and not device_status.get('no_cry_sent', False):
                            # Send no-cry notification
                            notification_sent = await send_nocry_notification({
                                'timestamp': current_time,
                                'deviceId': client_id,
                                'lastCryTimestamp': device_status['last_cry_time']
                            })

                            if notification_sent:
                                await self.manager.send_message(client_id, {
                                    "type": "alert",
                                    "timestamp": current_time,
                                    "message": "No crying detected",
                                    "deviceId": client_id
                                })
                                logger.info(
                                    f"No-cry event detected at {convert_utc_timestamp_to_vn_datetime(current_time)} for device {client_id}. "
                                    f"Last cry was at {device_status['last_cry_time']}, "
                                    f"{time_since_last_cry:.2f}s ago"
                                )
                                device_status['no_cry_sent'] = True

            except Exception as e:
                logger.error(f"Error in periodic check: {str(e)}")
            
            await asyncio.sleep(1)  # Wait for 1 second before next check

    def init_device_tracking(self, client_id: str):
        """Initialize or reset cry status tracking for a device"""
        self.cry_status_tracker[client_id] = {
            'last_cry_time': None,
            'no_cry_sent': False
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
        
        # Update last cry time and reset no_cry_sent flag
        device_status['last_cry_time'] = timestamp
        device_status['no_cry_sent'] = False

        # Send cry notification to Firebase
        notification_sent = await send_cry_notification({
            'timestamp': timestamp,
            'deviceId': client_id,
            'confidence': confidence
        })

        return notification_sent

    async def send_final_no_cry_events(self):
        """Send no-cry events for all tracked devices before shutdown"""
        current_time = asyncio.get_event_loop().time()
        for client_id, device_status in list(self.cry_status_tracker.items()):
            if device_status['last_cry_time'] is not None and not device_status.get('no_cry_sent', False):
                # Send no-cry notification for this device
                notification_sent = await send_nocry_notification({
                    'timestamp': current_time,
                    'deviceId': client_id,
                    'lastCryTimestamp': device_status['last_cry_time']
                })

                if notification_sent:
                    await self.manager.send_message(client_id, {
                        "type": "alert",
                        "timestamp": current_time,
                        "message": "No crying detected (server shutdown)",
                        "deviceId": client_id
                    })
                    logger.info(
                        f"Final no-cry event sent for device {client_id} during shutdown. "
                        f"Last cry was at {device_status['last_cry_time']}, "
                        f"{current_time - device_status['last_cry_time']:.2f}s ago"
                    )