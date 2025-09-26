# gui_module.py
import tkinter as tk
from tkinter import messagebox
import cv2

class UserInterface:
    def __init__(self):
        # Create the main GUI window
        self.root = tk.Tk()
        self.root.title("Media Player Control")
        self.root.geometry("300x200")
        
        # Play/Pause button
        self.play_pause_button = tk.Button(self.root, text="Play/Pause", command=self.play_pause)
        self.play_pause_button.pack(pady=10)

        # Volume Up button
        self.volume_up_button = tk.Button(self.root, text="Volume Up", command=self.volume_up)
        self.volume_up_button.pack(pady=10)

        # Volume Down button
        self.volume_down_button = tk.Button(self.root, text="Volume Down", command=self.volume_down)
        self.volume_down_button.pack(pady=10)

        # Exit button
        self.exit_button = tk.Button(self.root, text="Exit", command=self.exit_app)
        self.exit_button.pack(pady=10)

    def play_pause(self):
        messagebox.showinfo("Action", "Play/Pause Media")
        # Add your play/pause function here

    def volume_up(self):
        messagebox.showinfo("Action", "Increase Volume")
        # Add your volume up function here

    def volume_down(self):
        messagebox.showinfo("Action", "Decrease Volume")
        # Add your volume down function here

    def exit_app(self):
        self.root.quit()

    def update_display(self, frame, gesture_name):
        """
        Display the detected gesture on the OpenCV frame.
        
        Parameters:
            frame (numpy.ndarray): The video frame.
            gesture_name (str): The name of the detected gesture.
        """
        if gesture_name:
            cv2.putText(frame, f"Gesture: {gesture_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Media Player Control", frame)

