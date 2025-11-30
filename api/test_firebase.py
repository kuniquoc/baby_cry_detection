import firebase_service
import fcm_service
import logging
import asyncio

# Configure logging to show on console
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('firebase_test')

def test_firebase_connection():
    """Test if Firebase can be initialized properly"""
    logger.info("Testing Firebase connection...")
    
    # Attempt to initialize Firebase
    result = firebase_service.initialize_firebase()
    
    if result:
        logger.info("✅ SUCCESS: Firebase connection established successfully")
        return True
    else:
        logger.error("❌ FAILED: Could not connect to Firebase")
        return False

async def test_fcm_notification():
    """Test sending a simple FCM notification"""
    logger.info("Testing FCM notification...")
    
    # Test data
    test_token = "eJR67MaVToiRzpUnqflHug:APA91bHWVc4TbmESAYUj3_HtSFo4iNfmMTHHt8Q7As5MSI6KsQtIP1mxu9-cN6juBAlvL-XqQVibNpor6MNoQOZEKdq_6HuznwbK9kqTBetMoEl4vst7jOc"  # Replace with your actual test device FCM token
    test_title = "Test Notification"
    test_body = "This is a test notification"
    
    # Send test notification
    result = fcm_service.send_fcm_notification(
        token=test_token,
        title=test_title,
        body=test_body,
        data={"test": "true"}
    )
    
    if result:
        logger.info("✅ SUCCESS: FCM notification sent successfully")
        return True
    else:
        logger.error("❌ FAILED: Could not send FCM notification")
        return False

if __name__ == "__main__":
    # Run the Firebase connection test
    connection_success = test_firebase_connection()
    
    # Run the FCM notification test
    if connection_success:
        fcm_success = asyncio.run(test_fcm_notification())
    else:
        fcm_success = False
    
    # Show the final results
    print("\n" + "="*50)
    if connection_success:
        print("✅ Firebase connection test: SUCCESS")
        print("Firebase credentials file was found and the connection was established.")
    else:
        print("❌ Firebase connection test: FAILED")
        print("Please check the following:")
        print("1. Firebase credentials file exists at: api/firebase-credentials.json")
        print("2. The credentials file contains valid Firebase service account information")
        print("3. The Firebase project is properly set up and accessible")
    
    if fcm_success:
        print("\n✅ FCM notification test: SUCCESS")
        print("Test notification was sent successfully")
    else:
        print("\n❌ FCM notification test: FAILED")
        print("Could not send test notification")
    print("="*50)