# gui_dqn.py
"""
Simple Tkinter GUI to visualize training rewards saved by train_dqn.py
and to evaluate a saved model (models/dqn_cartpole.pth) for N episodes (greedy).
"""

import os
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

import gym
import torch
import torch.nn as nn

# ---------------------
# QNetwork (same architecture as train)
# ---------------------
class QNetwork(nn.Module):
    def __init__(self, obs_size, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        return self.net(x)

# ---------------------
# Gym helpers for compatibility
# ---------------------
def reset_env(env):
    out = env.reset()
    if isinstance(out, tuple):
        return out[0]
    return out

def step_env(env, action):
    out = env.step(action)
    if len(out) == 5:
        next_state, reward, terminated, truncated, info = out
        done = bool(terminated or truncated)
    else:
        next_state, reward, done, info = out
    return next_state, reward, done, info

# ---------------------
# GUI Application
# ---------------------
class DQNGui:
    def __init__(self, root):
        self.root = root
        root.title("DQN CartPole - Performance Viewer")
        self.model_path = os.path.join("models", "dqn_cartpole.pth")
        self.rewards_path = os.path.join("models", "rewards.npy")

        # Figure / plot init
        self.fig, self.ax = plt.subplots(figsize=(7,4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Controls
        ctrl_frame = ttk.Frame(root)
        ctrl_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=8)

        self.load_btn = ttk.Button(ctrl_frame, text="Load training plot", command=self.load_plot)
        self.load_btn.pack(side=tk.LEFT, padx=4)

        self.eval_btn = ttk.Button(ctrl_frame, text="Evaluate model (10 eps)", command=self.evaluate_model)
        self.eval_btn.pack(side=tk.LEFT, padx=4)

        self.choose_btn = ttk.Button(ctrl_frame, text="Choose model...", command=self.choose_model)
        self.choose_btn.pack(side=tk.LEFT, padx=4)

        self.clear_btn = ttk.Button(ctrl_frame, text="Reset plot", command=self.reset_plot)
        self.clear_btn.pack(side=tk.LEFT, padx=4)

        self.status = ttk.Label(root, text="Status: Ready")
        self.status.pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=4)

        # internal data
        self.training_rewards = []
        self.eval_rewards = []

        # initial plot load
        self.load_plot()

    def set_status(self, text):
        self.status.config(text=f"Status: {text}")
        self.root.update_idletasks()

    def load_plot(self):
        self.ax.clear()
        if os.path.exists(self.rewards_path):
            try:
                self.training_rewards = list(np.load(self.rewards_path))
                self.ax.plot(self.training_rewards, label="training reward")
            except Exception as e:
                self.set_status(f"Error loading rewards.npy: {e}")
                self.training_rewards = []
        else:
            self.set_status("No rewards.npy found (train first).")

        if self.eval_rewards:
            self.ax.plot(range(len(self.training_rewards), len(self.training_rewards) + len(self.eval_rewards)),
                         self.eval_rewards, label="evaluation reward", linestyle='--', marker='o')

        self.ax.set_xlabel("Episode")
        self.ax.set_ylabel("Reward")
        self.ax.set_title("Episode vs Reward")
        self.ax.grid(True)
        self.ax.legend()
        self.canvas.draw()
        self.set_status("Plot updated")

    def reset_plot(self):
        self.training_rewards = []
        self.eval_rewards = []
        self.ax.clear()
        self.canvas.draw()
        self.set_status("Plot reset")

    def choose_model(self):
        path = filedialog.askopenfilename(title="Choose model file",
                                          filetypes=[("PyTorch model", "*.pth *.pt"), ("All files", "*.*")])
        if path:
            self.model_path = path
            self.set_status(f"Model set: {os.path.basename(path)}")

    def evaluate_model(self):
        if not os.path.exists(self.model_path):
            messagebox.showerror("Model not found", f"Model not found at {self.model_path}\nRun training first or choose a model.")
            return

        self.set_status("Loading model...")
        env = gym.make("CartPole-v1")
        obs_size = env.observation_space.shape[0]
        n_actions = env.action_space.n

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = QNetwork(obs_size, n_actions).to(device)
        model.load_state_dict(torch.load(self.model_path, map_location=device))
        model.eval()

        N = 10
        eval_rewards = []
        self.set_status("Evaluating (greedy)...")
        for ep in range(N):
            state = reset_env(env)
            done = False
            total_r = 0.0
            steps = 0
            while not done:
                with torch.no_grad():
                    s_v = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                    action = int(torch.argmax(model(s_v), dim=1).item())
                next_state, r, done, _ = step_env(env, action)
                total_r += r
                state = next_state
                steps += 1
                if steps > 1000:
                    break
            eval_rewards.append(total_r)
            self.set_status(f"Eval: episode {ep+1}/{N} reward {total_r:.1f}")

        env.close()
        # append to eval_rewards and redraw
        self.eval_rewards = eval_rewards
        self.load_plot()
        avg = float(np.mean(eval_rewards))
        messagebox.showinfo("Evaluation finished", f"Ran {N} episodes (greedy).\nAverage reward: {avg:.2f}")
        self.set_status("Evaluation finished")

if __name__ == "__main__":
    root = tk.Tk()
    app = DQNGui(root)
    root.mainloop()
