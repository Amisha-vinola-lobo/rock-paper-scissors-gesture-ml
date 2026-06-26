import cv2
import tensorflow as tf
import numpy as np
import random
import time
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

# Load model
model = tf.keras.models.load_model('model10_finetuned.h5')
class_names = ['paper', 'rock', 'scissor']

cap = cv2.VideoCapture(0)

rounds = 0
current_round = 0
player_score = 0
computer_score = 0

is_running = False
countdown_active = False
countdown_end = 0
last_frame = None

root = tk.Tk()
root.title("Rock Paper Scissors")
root.configure(bg="#2b0871")

title_label = tk.Label(root, text="Rock Paper Scissors", font=("Segoe UI", 28, "bold"),
                       fg="#f9fafb", bg="#2b0871")
title_label.pack(pady=10)

top_frame = tk.Frame(root, bg="#2b0871")
top_frame.pack(pady=10)

round_label = tk.Label(top_frame, text="Number of rounds:", font=("Segoe UI", 14),
                       fg="#e5e7eb", bg="#2b0871")
round_label.pack(side="left", padx=(0, 8))

round_entry = tk.Entry(top_frame, width=6, font=("Segoe UI", 14))
round_entry.insert(0, "3")
round_entry.pack(side="left", padx=(0, 12))

start_button = tk.Button(top_frame, text="Start game", font=("Segoe UI", 14, "bold"),
                         bg="#10b981", fg="white", activebackground="#0ea5e9",
                         command=lambda: start_game())
start_button.pack(side="left")

game_frame = tk.Frame(root, bg="#2b0871")
game_frame.pack(pady=20)

player_label = tk.Label(game_frame, bg="black", width=480, height=360)
player_label.grid(row=0, column=0, padx=20)

vs_label = tk.Label(game_frame, text="VS", font=("Segoe UI", 26, "bold"),
                    fg="#f9fafb", bg="#2b0871")
vs_label.grid(row=0, column=1, padx=10)

computer_box = tk.Frame(game_frame, bg="#111827")
computer_box.grid(row=0, column=2, padx=20)
computer_title = tk.Label(computer_box, text="Computer", font=("Segoe UI", 18, "bold"),
                          fg="#f9fafb", bg="#111827")
computer_title.pack(pady=(12, 6), padx=12)
computer_label = tk.Label(computer_box, text="—", font=("Segoe UI", 20),
                          fg="#93c5fd", bg="#111827")
computer_label.pack(pady=(0, 12), padx=12)

# NEW: player choice text
player_choice_label = tk.Label(root, text="Your choice: —", font=("Segoe UI", 16),
                               fg="#93c5fd", bg="#2b0871")
player_choice_label.pack(pady=(0, 6))

score_label = tk.Label(root, text="Player: 0 | Computer: 0", font=("Segoe UI", 16),
                       fg="#f9fafb", bg="#2b0871")
score_label.pack(pady=(6, 0))

result_label = tk.Label(root, text="", font=("Segoe UI", 18, "bold"),
                        fg="#fcd34d", bg="#2b0871")
result_label.pack(pady=(6, 12))

cap = cv2.VideoCapture(0)

def decide_winner(player, computer):
    if player == computer:
        return "Draw"
    wins = (player == "rock" and computer == "scissor") or \
           (player == "paper" and computer == "rock") or \
           (player == "scissor" and computer == "paper")
    return "Player" if wins else "Computer"

def start_countdown():
    global countdown_active, countdown_end
    countdown_active = True
    countdown_end = time.time() + 3

def start_game():
    global rounds, current_round, player_score, computer_score, is_running
    try:
        r = int(round_entry.get())
        if r <= 0:
            raise ValueError
        rounds = r
    except:
        messagebox.showerror("Invalid input", "Please enter a valid positive integer for rounds.")
        return

    current_round = 0
    player_score = 0
    computer_score = 0
    score_label.config(text=f"Player: {player_score} | Computer: {computer_score}")
    result_label.config(text="")
    computer_label.config(text="—")
    player_choice_label.config(text="Your choice: —")

    is_running = True
    start_countdown()

def end_game():
    global is_running
    is_running = False

    if player_score > computer_score:
        final_winner = "Player"
    elif computer_score > player_score:
        final_winner = "Computer"
    else:
        final_winner = "Draw"

    result_label.config(text=f"Final winner: {final_winner}")
    answer = messagebox.askyesno(
        "Game over",
        f"Final score\nPlayer: {player_score} | Computer: {computer_score}\n\n"
        f"Winner: {final_winner}\n\nContinue game?"
    )
    if answer:
        start_game()
    else:
        cap.release()
        root.destroy()

def process_round(frame_bgr):
    global player_score, computer_score, current_round

    img = cv2.resize(frame_bgr, (100, 100))
    img_array = np.expand_dims(img, axis=0) / 255.0
    preds = model.predict(img_array)
    player_choice = class_names[int(np.argmax(preds))]

    computer_choice = random.choice(class_names)
    computer_label.config(text=computer_choice)
    player_choice_label.config(text=f"Your choice: {player_choice}")  # NEW

    winner = decide_winner(player_choice, computer_choice)
    if winner == "Player":
        player_score += 1
    elif winner == "Computer":
        computer_score += 1

    score_label.config(text=f"Player: {player_score} | Computer: {computer_score}")
    result_label.config(text=f"Round {current_round + 1} winner: {winner}")

    current_round += 1
    if current_round >= rounds:
        end_game()
    else:
        # NEW: delay before next round
        root.after(2000, start_countdown)  # 2 second pause

def update_frame():
    global last_frame, countdown_active

    ret, frame_bgr = cap.read()
    if ret:
        last_frame = frame_bgr.copy()
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        if is_running and countdown_active:
            remaining = int(round(countdown_end - time.time()))
            if remaining > 0:
                h, w, _ = frame_rgb.shape
                cv2.putText(frame_rgb, str(remaining), (w // 2 - 40, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 80, 80), 8)
            else:
                countdown_active = False
                if last_frame is not None:
                    process_round(last_frame)

        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        player_label.imgtk = imgtk
        player_label.configure(image=imgtk)

    root.after(20, update_frame)

update_frame()
root.mainloop()