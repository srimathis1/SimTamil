"""
chat_ui.py -- Built-in Tkinter Desktop Chatbot Interface.

Opens a native GUI chat window immediately upon execution.
Loads latent memory from data/Expert_reflection/ and runs via Gemini.

Usage:
    python scripts/chat_ui.py
"""

import os
import sys
import threading
from pathlib import Path
import tkinter as tk
from tkinter import ttk, scrolledtext
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("API_KEY")
# Setup API Key
os.environ["GEMINI_API_KEY"] = api_key
genai.configure(api_key=api_key)

MODEL_NAME = "gemini-3.6-flash"


def load_grounded_memory(refl_dir: Path) -> str:
    """Reads all expert reflection text files from data/Expert_reflection/."""
    expected_files = [
        "psychologist_reflection.txt",
        "behavioral_economist_reflection.txt",
        "political_scientist_reflection.txt",
        "demographer_reflection.txt",
        "big_five_scores.txt",
    ]
    parts = []
    for fname in expected_files:
        fpath = refl_dir / fname
        if fpath.exists():
            parts.append(f"=== {fname.upper()} ===\n{fpath.read_text(encoding='utf-8').strip()}")

    if not parts:
        print(f"Warning: No reflection files found in {refl_dir}.")
    return "\n\n".join(parts)


class GroundedChatApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Tamil Grounded Persona Chatbot")
        self.root.geometry("640x720")
        self.root.minsize(500, 600)
        self.root.configure(bg="#F4F6F8")

        # Load Persona Memory
        project_root = Path(__file__).resolve().parent.parent
        refl_dir = project_root / "data" / "Expert_reflection"
        self.latent_memory = load_grounded_memory(refl_dir)

        system_instruction = f"""
You are an autonomous behavioral clone and digital twin of the participant profiled in the memory below.

=== LATENT MEMORY & EXPERT PROFILES ===
{self.latent_memory}

=== BEHAVIORAL DIRECTIVES ===
1. First-Person Persona: Always speak as 'I' (நான்). Never break character or refer to yourself as an AI.
2. Authentic Conversational Tamil: Speak in natural, everyday conversational Tamil using authentic slang, natural flow, and standard English loanwords (e.g., college, tension, family, savings) consistent with the audio interviews.
3. Decision & Trait Fidelity: Strictly adhere to your verified Big Five traits, financial risk bounds, and demographic constraints when answering new or hypothetical questions.
"""
        model = genai.GenerativeModel(
            model_name=MODEL_NAME,
            system_instruction=system_instruction,
        )
        self.chat = model.start_chat(history=[])

        self._build_ui()
        self._append_message("Agent", "வணக்கம்! என்ன விஷயம் பேசலாம்?")

    def _build_ui(self):
        # Header banner
        header_frame = tk.Frame(self.root, bg="#1E293B", height=60)
        header_frame.pack(fill=tk.X)

        title_lbl = tk.Label(
            header_frame,
            text="Grounded Tamil Persona Agent",
            font=("Segoe UI", 13, "bold"),
            fg="#FFFFFF",
            bg="#1E293B",
        )
        title_lbl.pack(pady=(12, 2))

        sub_lbl = tk.Label(
            header_frame,
            text="Calibrated against Tamil interviews & Big Five surveys",
            font=("Segoe UI", 9),
            fg="#94A3B8",
            bg="#1E293B",
        )
        sub_lbl.pack(pady=(0, 10))

        # Chat display area
        chat_frame = tk.Frame(self.root, bg="#F4F6F8")
        chat_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)

        self.chat_display = scrolledtext.ScrolledText(
            chat_frame,
            wrap=tk.WORD,
            font=("Segoe UI", 10),
            bg="#FFFFFF",
            fg="#1E293B",
            relief=tk.FLAT,
            padx=12,
            pady=12,
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True)
        self.chat_display.config(state=tk.DISABLED)

        # Style tags for messages
        self.chat_display.tag_config("user_tag", foreground="#2563EB", font=("Segoe UI", 10, "bold"))
        self.chat_display.tag_config("agent_tag", foreground="#059669", font=("Segoe UI", 10, "bold"))
        self.chat_display.tag_config("body_tag", foreground="#1E293B", font=("Segoe UI", 10))

        # Bottom Input Area
        input_frame = tk.Frame(self.root, bg="#F4F6F8")
        input_frame.pack(fill=tk.X, padx=15, pady=(0, 15))

        self.entry_box = tk.Entry(
            input_frame,
            font=("Segoe UI", 11),
            bg="#FFFFFF",
            fg="#1E293B",
            relief=tk.SOLID,
            bd=1,
        )
        self.entry_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, ipady=8, padx=(0, 10))
        self.entry_box.bind("<Return>", lambda event: self.send_message())
        self.entry_box.focus()

        self.send_btn = tk.Button(
            input_frame,
            text="Send",
            font=("Segoe UI", 10, "bold"),
            bg="#2563EB",
            fg="#FFFFFF",
            activebackground="#1D4ED8",
            activeforeground="#FFFFFF",
            relief=tk.FLAT,
            padx=20,
            command=self.send_message,
        )
        self.send_btn.pack(side=tk.RIGHT)

    def _append_message(self, sender: str, text: str):
        self.chat_display.config(state=tk.NORMAL)
        if sender == "You":
            self.chat_display.insert(tk.END, "You: ", "user_tag")
        else:
            self.chat_display.insert(tk.END, "Agent: ", "agent_tag")

        self.chat_display.insert(tk.END, f"{text}\n\n", "body_tag")
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)

    def send_message(self):
        user_text = self.entry_box.get().strip()
        if not user_text:
            return

        self.entry_box.delete(0, tk.END)
        self._append_message("You", user_text)

        # Disable send button while waiting
        self.send_btn.config(state=tk.DISABLED)

        # Run Gemini call in a separate thread to keep UI smooth
        threading.Thread(target=self._query_agent, args=(user_text,), daemon=True).start()

    def _query_agent(self, prompt: str):
        try:
            response = self.chat.send_message(prompt)
            reply = response.text.strip()
        except Exception as e:
            reply = f"[Error: {e}]"

        self.root.after(0, lambda: self._on_reply_received(reply))

    def _on_reply_received(self, reply: str):
        self._append_message("Agent", reply)
        self.send_btn.config(state=tk.NORMAL)
        self.entry_box.focus()


def main():
    root = tk.Tk()
    app = GroundedChatApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()