# updated_integrated_main.py - FIXED VERSION
import cv2
import numpy as np
import tensorflow as tf
import pyautogui as p
import mediapipe as mp
import speech_recognition as sr
import threading
import time
import json
from queue import Queue
from collections import deque, Counter

# Load the newly trained model and its info
try:
    gesture_model = tf.keras.models.load_model('best_model.h5')
    print("✅ New gesture model loaded successfully")
    
    # Load model info
    with open('model_info.json', 'r') as f:
        model_info = json.load(f)
    
    gesture_labels = model_info['label_names']
    gesture_mapping = model_info['gesture_mapping']
    
    print(f"📋 Loaded gesture mapping: {gesture_mapping}")
    print(f"🏷️ Label order: {gesture_labels}")
    
except FileNotFoundError:
    print("❌ Could not load new model files!")
    print("Please run the training script first to create:")
    print("  - gesture_model_final.h5")
    print("  - model_info.json")
    exit(1)
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

# Voice command labels
voice_labels = ['play', 'pause', 'next', 'previous', 'volume_up', 'volume_down']

# Initialize MediaPipe Hands - RELAXED SETTINGS
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,  # REDUCED from 0.8
    min_tracking_confidence=0.5    # REDUCED from 0.8
)

# Initialize Speech Recognition
recognizer = sr.Recognizer()
microphone = sr.Microphone()

# Global variables
voice_active = False
voice_command_queue = Queue()
gesture_active = True
cooldown = 0

    
# RELAXED gesture tracking parameters with STABILITY DELAY
gesture_buffer = deque(maxlen=8)  # Increased for better stability
confidence_threshold = 0.7  # REDUCED from 0.85
stable_count_needed = 6     # INCREASED for stability delay
min_stability_ratio = 0.75  # INCREASED for consistency
gesture_hold_frames = 0     # Track how long gesture is held
min_hold_duration = 15      # Minimum frames to hold gesture (0.5 seconds at 30fps)

class VoiceListener:
    def __init__(self):
        self.is_listening = False
        self.activation_phrases = ["hey windows", "hello windows", "windows"]
        
    def listen_for_activation(self):
        """Continuously listen for activation phrase"""
        global voice_active
        
        with microphone as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)
            
        while True:
            try:
                with microphone as source:
                    audio = recognizer.listen(source, timeout=1, phrase_time_limit=3)
                
                text = recognizer.recognize_google(audio).lower()
                print(f"Heard: {text}")
                
                for phrase in self.activation_phrases:
                    if phrase in text:
                        print("🎤 Voice control activated!")
                        voice_active = True
                        self.listen_for_command()
                        break
                        
            except sr.WaitTimeoutError:
                pass
            except sr.UnknownValueError:
                pass
            except sr.RequestError as e:
                print(f"Speech recognition error: {e}")
                time.sleep(1)
    
    def listen_for_command(self):
        """Listen for voice commands after activation"""
        global voice_active, voice_command_queue
        
        print("🎤 Listening for voice command...")
        try:
            with microphone as source:
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=3)
            
            text = recognizer.recognize_google(audio).lower()
            print(f"Command heard: {text}")
            
            command = self.process_voice_command(text)
            if command:
                voice_command_queue.put(command)
                
        except sr.WaitTimeoutError:
            print("⏰ Voice command timeout")
        except sr.UnknownValueError:
            print("❌ Could not understand command")
        except sr.RequestError as e:
            print(f"Speech recognition error: {e}")
        
        voice_active = False
    
    def process_voice_command(self, text):
        """Process recognized text into commands"""
        command_mapping = {
            'play': ['play', 'start', 'resume'],
            'pause': ['pause', 'stop', 'halt'],
            'next': ['next', 'skip', 'forward'],
            'previous': ['previous', 'back', 'backward', 'last'],
            'volume_up': ['volume up', 'louder', 'increase volume'],
            'volume_down': ['volume down', 'quieter', 'decrease volume', 'lower volume']
        }
        
        for command, keywords in command_mapping.items():
            for keyword in keywords:
                if keyword in text:
                    return command
        return None

def preprocess_for_new_model(hand_roi):
    """Preprocess image exactly as the training pipeline does"""
    try:
        if hand_roi is None or hand_roi.size == 0:
            return np.zeros((64, 64), dtype=np.float32)
        
        # Convert to grayscale if needed
        if len(hand_roi.shape) == 3:
            gray = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = hand_roi.copy()
        
        # Resize to model input size
        img = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
        
        # Apply threshold to create clean binary image
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        
        # Check if background is mostly white (need to invert)
        white_pixels = np.sum(binary == 255)
        black_pixels = np.sum(binary == 0)
        
        if white_pixels > black_pixels:
            # Invert to get white hand on black background
            binary = cv2.bitwise_not(binary)
        
        # Apply morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Normalize to [0, 1] as done in training
        normalized = binary.astype(np.float32) / 255.0
        
        return normalized
        
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return np.zeros((64, 64), dtype=np.float32)

def predict_gesture_with_confidence(hand_image):
    """Predict gesture using the newly trained model"""
    try:
        # Preprocess exactly as training
        processed_img = preprocess_for_new_model(hand_image)
        
        # Add batch and channel dimensions
        img_input = np.expand_dims(processed_img, axis=0)
        img_input = np.expand_dims(img_input, axis=-1)
        
        # Get prediction
        predictions = gesture_model.predict(img_input, verbose=0)[0]
        
        # Get top prediction
        predicted_idx = np.argmax(predictions)
        confidence = float(predictions[predicted_idx])
        predicted_gesture = gesture_labels[predicted_idx]
        
        # Create full probability distribution for debugging
        prob_dict = {gesture_labels[i]: float(predictions[i]) for i in range(len(gesture_labels))}
        
        return predicted_gesture, confidence, prob_dict
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        return "error", 0.0, {}

def get_stable_gesture():
    """Get stable gesture prediction with STABILITY DELAY"""
    global gesture_hold_frames
    
    if len(gesture_buffer) < stable_count_needed:
        gesture_hold_frames = 0  # Reset if buffer not full
        return None, 0.0, {}
    
    # Consider moderate confidence predictions
    valid_predictions = []
    for gesture, confidence, _ in gesture_buffer:
        if confidence >= confidence_threshold and gesture not in ["error"]:
            valid_predictions.append(gesture)
    
    if len(valid_predictions) < stable_count_needed:
        gesture_hold_frames = 0  # Reset if not enough valid predictions
        return None, 0.0, {}
    
    # Count occurrences
    gesture_counts = Counter(valid_predictions)
    
    if not gesture_counts:
        gesture_hold_frames = 0
        return None, 0.0, {}
    
    most_common_gesture, count = gesture_counts.most_common(1)[0]
    stability_ratio = count / len(valid_predictions)
    
    # Check if gesture is stable enough
    if stability_ratio >= min_stability_ratio and count >= stable_count_needed:
        # Get average confidence for this gesture
        confidences = [conf for gest, conf, _ in gesture_buffer if gest == most_common_gesture]
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        # Check if this is the same gesture as before (continuity)
        recent_gestures = [gest for gest, conf, _ in list(gesture_buffer)[-3:] if conf >= confidence_threshold]
        
        if len(recent_gestures) >= 2 and all(g == most_common_gesture for g in recent_gestures):
            # Same gesture detected consistently, increment hold counter
            gesture_hold_frames += 1
        else:
            # Different gesture or inconsistent, reset counter
            gesture_hold_frames = 0
        
        # Only return gesture if held long enough
        if gesture_hold_frames >= min_hold_duration:
            # Get the most recent probability distribution
            recent_probs = {}
            for gesture, confidence, probs in reversed(gesture_buffer):
                if gesture == most_common_gesture and probs:
                    recent_probs = probs
                    break
            
            return most_common_gesture, avg_confidence, recent_probs
    else:
        gesture_hold_frames = 0  # Reset if not stable
    
    return None, 0.0, {}

def execute_media_command(command):
    """Execute media control command with REDUCED cooldown"""
    global cooldown, last_command_executed, command_display_counter

    if cooldown > 0:
        return False  # Skip if still in cooldown

    print(f"EXECUTING: {command}")
    
    # Send corresponding key press
    if command == "play":
        p.press("playpause")
    elif command == "pause":
        p.press("space")
    elif command == "next":
        p.press("right")
    elif command == "previous":
        p.press("left")
    elif command == "volume_up":
        p.press("volumeup")
    elif command == "volume_down":
        p.press("volumedown")
    elif command == "stop":
        p.press("stop")

    # Update frame display command text
    last_command_executed = command.upper()          # You can use gesture_mapping[gesture] for labels if needed
    command_display_counter = 60                     # Show for 60 frames (~2 seconds)

    print(f"✅ Command {command} executed successfully!")
    cooldown = 30  # Set cooldown
    return True

    
    if cooldown > 0:
        print(f"⏳ Command blocked by cooldown: {cooldown} frames remaining")
        return False
    
    # Display gesture-to-finger mapping for user reference
    finger_mapping = {
        # Command names (what should be executed)
        'play': '1 finger',
        'next': '2 fingers', 
        'previous': '3 fingers',
        'volume_down': '4 fingers',
        'volume_up': '5 fingers',
        'pause': 'closed fist',
        # Gesture names (what model might return directly)
        'one_finger': '1 finger',
        'two_fingers': '2 fingers',
        'three_fingers': '3 fingers', 
        'four_fingers': '4 fingers',
        'five_fingers': '5 fingers',
        'closed_fist': 'closed fist',
        'fist': 'closed fist'
    }
    
    print(f"🎵 EXECUTING: {command} ({finger_mapping.get(command, 'unknown gesture')})")
    
    try:
        # Multiple key strategies for better compatibility
        if command in ['play', 'one_finger']:  # 1 finger - PLAY
            print("▶️ Sending PLAY command...")
            p.press('space')  # Most common
            time.sleep(0.1)
            p.press('playpause')  # Media key
            cooldown = 20  # REDUCED from 60
            
        elif command in ['pause', 'closed_fist', 'fist']:  # closed fist - PAUSE
            print("⏸️ Sending PAUSE command...")
            p.press('space')  # Most common
            time.sleep(0.1) 
            p.press('playpause')  # Media key
            time.sleep(0.1)
            p.press('pause')  # Direct pause key
            cooldown = 20  # REDUCED from 60
            
        elif command in ['next', 'two_fingers']:  # 2 fingers - NEXT
            print("⏭️ Sending NEXT command...")
            p.press('right')  # Arrow key
            time.sleep(0.1)
            p.press('nexttrack')  # Media key
            time.sleep(0.1)
            p.hotkey('ctrl', 'right')  # Alternative
            cooldown = 20  # REDUCED from 60
            
        elif command in ['previous', 'three_fingers']:  # 3 fingers - PREVIOUS
            print("⏮️ Sending PREVIOUS command...")
            p.press('left')  # Arrow key
            time.sleep(0.1)
            p.press('prevtrack')  # Media key
            time.sleep(0.1)
            p.hotkey('ctrl', 'left')  # Alternative
            cooldown = 20  # REDUCED from 60
            
        elif command in ['volume_up', 'five_fingers']:  # 5 fingers - VOLUME UP
            print("🔊 Sending VOLUME UP command...")
            p.press('up')  # Arrow key
            time.sleep(0.1)
            p.press('volumeup')  # Media key
            time.sleep(0.1)
            p.hotkey('ctrl', 'up')  # Alternative
            cooldown = 10  # REDUCED from 30
            
        elif command in ['volume_down', 'four_fingers']:  # 4 fingers - VOLUME DOWN
            print("🔉 Sending VOLUME DOWN command...")
            p.press('down')  # Arrow key
            time.sleep(0.1)
            p.press('volumedown')  # Media key
            time.sleep(0.1)
            p.hotkey('ctrl', 'down')  # Alternative
            cooldown = 10  # REDUCED from 30
            
        else:
            print(f"❌ Unknown command: {command}")
            return False
            
        print(f"✅ Command {command} executed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error executing command {command}: {e}")
        return False

def main():
    global voice_active, cooldown, gesture_buffer, gesture_active, gesture_hold_frames
    
    print("🚀 FIXED GESTURE CONTROL SYSTEM")
    print("=" * 60)
    print("🖐️ GESTURE MAPPING:")
    print("  👆 1 finger    = Play")
    print("  ✌️  2 fingers   = Next")
    print("  🤟 3 fingers   = Previous")
    print("  🖖 4 fingers   = Volume Down")
    print("  🖐️ 5 fingers   = Volume Up")
    print("  ✊ Closed fist = Pause")
    print("=" * 60)
    print("🔧 FIXES APPLIED:")
    print("  ✅ Reduced detection confidence: 0.8 → 0.5")
    print("  ✅ Reduced prediction confidence: 0.85 → 0.7") 
    print("  ✅ Added gesture stability delay: 15 frames (0.5s)")
    print("  ✅ Improved buffer requirements: 6/8 stable predictions")
    print("  ✅ Reduced cooldown: 60/30 → 20/10 frames")
    print("  ✅ Process every frame (removed frame skipping)")
    print("=" * 60)
    print("🎤 Voice: Say 'Hey Windows' for voice control")
    print("⌨️  Controls: 'q'=quit, 'r'=reset buffer, 'c'=calibration mode")
    print("🧪 Manual Test: Press 1-5,0 to test commands directly")
    print("=" * 60)
    
    # Start voice listener
    voice_listener = VoiceListener()
    voice_thread = threading.Thread(target=voice_listener.listen_for_activation, daemon=True)
    voice_thread.start()
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    frame_count = 0
    calibration_mode = False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        frame = cv2.flip(frame, 1)
        
        # Reduce cooldown FASTER
        if cooldown > 0:
            cooldown -= 2  # FASTER cooldown reduction
        
        # Process voice commands
        if not voice_command_queue.empty():
            voice_command = voice_command_queue.get()
            execute_media_command(voice_command)
        
        # Convert for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        # Get frame dimensions early
        h, w, _ = frame.shape
        
        current_gesture = "No hand detected"
        confidence = 0.0
        prob_distribution = {}
        
        # Process hands EVERY FRAME for better detection
        if results.multi_hand_landmarks and gesture_active:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw enhanced hand landmarks
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=3)
                )
                
                # Extract hand region with optimal padding
                x_coords = [int(lm.x * w) for lm in hand_landmarks.landmark]
                y_coords = [int(lm.y * h) for lm in hand_landmarks.landmark]
                
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                
                # Add generous padding
                padding = 50
                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)
                x_max = min(w, x_max + padding)
                y_max = min(h, y_max + padding)
                
                # Extract hand ROI
                hand_roi = frame[y_min:y_max, x_min:x_max]
                
                if hand_roi.size > 0:
                    # Predict gesture
                    gesture, conf, probs = predict_gesture_with_confidence(hand_roi)
                    
                    # Add to buffer
                    gesture_buffer.append((gesture, conf, probs))
                    
                    # Get stable prediction
                    stable_gesture, stable_conf, stable_probs = get_stable_gesture()
                    
                    if stable_gesture:
                        current_gesture = stable_gesture
                        confidence = stable_conf
                        prob_distribution = stable_probs
                        
                        # Execute command if confidence is high enough AND held long enough
                        if stable_conf >= confidence_threshold and cooldown == 0:
                            # First try to map gesture to command using gesture_mapping
                            command = None
                            if stable_gesture in gesture_mapping:
                                command = gesture_mapping[stable_gesture]
                                print(f"🔄 Mapped {stable_gesture} -> {command}")
                            else:
                                # Fallback: use gesture name directly
                                command = stable_gesture
                                print(f"🔄 Using gesture name directly: {command}")
                            
                            if command and execute_media_command(command):
                                gesture_buffer.clear()  # Clear buffer after successful command
                                gesture_hold_frames = 0  # Reset hold counter
                    
                    # Draw bounding box around hand
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    
                    # Show preprocessed hand in calibration mode
                    if calibration_mode and hand_roi.size > 0:
                        processed = preprocess_for_new_model(hand_roi)
                        processed_display = (processed * 255).astype(np.uint8)
                        processed_display = cv2.resize(processed_display, (200, 200))
                        processed_colored = cv2.applyColorMap(processed_display, cv2.COLORMAP_JET)
                        
                        # Show processed image
                        cv2.imshow('Processed Hand', processed_colored)
        
        # Create info display
        info_y = 30
        line_height = 30
        
        # Status indicators
        status_color = (0, 255, 0) if gesture_active else (0, 0, 255)
        cv2.putText(frame, f"Gesture: {'ON' if gesture_active else 'OFF'}", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        info_y += line_height
        
        voice_color = (255, 255, 0) if voice_active else (128, 128, 128)
        cv2.putText(frame, f"Voice: {'LISTENING' if voice_active else 'STANDBY'}", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, voice_color, 2)
        info_y += line_height
        
        # Current gesture and confidence
        gesture_color = (0, 255, 255) if confidence >= confidence_threshold else (0, 165, 255)
        cv2.putText(frame, f"Gesture: {current_gesture}", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, gesture_color, 2)
        info_y += line_height
        
        cv2.putText(frame, f"Confidence: {confidence:.2f}", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, gesture_color, 2)
        info_y += line_height
        
        # Cooldown indicator
        if cooldown > 0:
            cv2.putText(frame, f"Cooldown: {cooldown}", 
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            info_y += line_height
        
        # Buffer status with hold duration
        buffer_text = f"Buffer: {len(gesture_buffer)}/{stable_count_needed}"
        cv2.putText(frame, buffer_text, 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        info_y += line_height
        
        # Hold duration indicator
        if gesture_hold_frames > 0:
            hold_progress = min(gesture_hold_frames / min_hold_duration, 1.0)
            hold_color = (0, 255, 0) if hold_progress >= 1.0 else (0, 165, 255)
            hold_text = f"Hold: {gesture_hold_frames}/{min_hold_duration} ({hold_progress:.1%})"
            cv2.putText(frame, hold_text, 
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, hold_color, 2)
            info_y += line_height
        
        # Show probability distribution in calibration mode
        if calibration_mode and prob_distribution:
            prob_y = info_y + 20
            cv2.putText(frame, "Probabilities:", 
                       (10, prob_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            prob_y += 25
            
            # Sort by probability for better display
            sorted_probs = sorted(prob_distribution.items(), key=lambda x: x[1], reverse=True)
            for i, (label, prob) in enumerate(sorted_probs[:6]):  # Show top 6
                color = (0, 255, 0) if prob >= confidence_threshold else (255, 255, 255)
                cv2.putText(frame, f"{label}: {prob:.3f}", 
                           (10, prob_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                prob_y += 20        

        # Instructions
        instructions = [
            "Controls:",
            "Q - Quit",
            "R - Reset buffer", 
            "C - Toggle calibration",
            "G - Toggle gesture mode",
            "1-5,0 - Test commands manually"
        ]
        
        inst_y = h - len(instructions) * 25 - 10
        for instruction in instructions:
            cv2.putText(frame, instruction, 
                       (10, inst_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            inst_y += 25
            
        
        # Show frame
        cv2.imshow('Gesture Control System', frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            gesture_buffer.clear()
            gesture_hold_frames = 0
            print("🔄 Buffer and hold counter reset")
        elif key == ord('c'):
            calibration_mode = not calibration_mode
            print(f"🔧 Calibration mode: {'ON' if calibration_mode else 'OFF'}")
        elif key == ord('g'):
            gesture_active = not gesture_active
            print(f"🖐️ Gesture recognition: {'ON' if gesture_active else 'OFF'}")
        # Manual testing keys
        elif key == ord('1'):
            print("🧪 Testing PLAY command manually...")
            execute_media_command('play')
        elif key == ord('2'):
            print("🧪 Testing NEXT command manually...")
            execute_media_command('next')
        elif key == ord('3'):
            print("🧪 Testing PREVIOUS command manually...")
            execute_media_command('previous')
        elif key == ord('4'):
            print("🧪 Testing VOLUME DOWN command manually...")
            execute_media_command('volume_down')
        elif key == ord('5'):
            print("🧪 Testing VOLUME UP command manually...")
            execute_media_command('volume_up')
        elif key == ord('0'):
            print("🧪 Testing PAUSE command manually...")
            execute_media_command('pause')
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("👋 Gesture control system stopped")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Program interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()