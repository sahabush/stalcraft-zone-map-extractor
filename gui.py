#!/usr/bin/env python3
"""
Simple one-window GUI for stalcraft-zone-map-extractor.

Does the whole job with a few clicks — no command line needed:
  * install dependencies
  * pick a map_cache location folder
  * choose mode (roofs / xray) and scale
  * render, watch the log, open the result

Uses only the Python standard library (tkinter). Run:  python gui.py
"""
import os
import sys
import threading
import subprocess
import queue
import tkinter as tk
from tkinter import ttk, filedialog

HERE = os.path.dirname(os.path.abspath(__file__))
SCMAP = os.path.join(HERE, "scmap.py")
REQS = os.path.join(HERE, "requirements.txt")


class App:
    def __init__(self, root):
        root.title("STALCRAFT Zone Map Extractor")
        root.geometry("720x560")
        self.root = root
        self.q = queue.Queue()
        self.busy = False

        pad = dict(padx=10, pady=6)

        # --- map folder ---
        frm = ttk.LabelFrame(root, text="1. Папка локации (map_cache/5.0/<локация>)")
        frm.pack(fill="x", **pad)
        self.path_var = tk.StringVar()
        ttk.Entry(frm, textvariable=self.path_var).pack(side="left", fill="x", expand=True, padx=8, pady=8)
        ttk.Button(frm, text="Обзор…", command=self.browse).pack(side="left", padx=8)

        # --- options ---
        opt = ttk.LabelFrame(root, text="2. Настройки")
        opt.pack(fill="x", **pad)
        self.mode = tk.StringVar(value="roofs")
        ttk.Label(opt, text="Режим:").grid(row=0, column=0, sticky="w", padx=8, pady=8)
        ttk.Radiobutton(opt, text="roofs (карта с крышами)", variable=self.mode, value="roofs").grid(row=0, column=1, sticky="w")
        ttk.Radiobutton(opt, text="xray (планировка зданий)", variable=self.mode, value="xray").grid(row=0, column=2, sticky="w")
        ttk.Label(opt, text="Масштаб (px/блок):").grid(row=1, column=0, sticky="w", padx=8, pady=8)
        self.scale = tk.IntVar(value=3)
        ttk.Spinbox(opt, from_=1, to=10, width=5, textvariable=self.scale).grid(row=1, column=1, sticky="w")
        self.outdir = os.path.join(HERE, "maps")

        # --- buttons ---
        btns = ttk.Frame(root)
        btns.pack(fill="x", **pad)
        self.btn_deps = ttk.Button(btns, text="Установить зависимости", command=self.install_deps)
        self.btn_deps.pack(side="left", padx=8)
        self.btn_run = ttk.Button(btns, text="Создать карту", command=self.run_render)
        self.btn_run.pack(side="left", padx=8)
        self.btn_open = ttk.Button(btns, text="Открыть результаты", command=self.open_out)
        self.btn_open.pack(side="left", padx=8)

        # --- log ---
        logf = ttk.LabelFrame(root, text="Лог")
        logf.pack(fill="both", expand=True, **pad)
        self.log = tk.Text(logf, height=14, bg="#15150C", fg="#EAE6D8", insertbackground="#EAE6D8", wrap="word")
        self.log.pack(side="left", fill="both", expand=True, padx=(8, 0), pady=8)
        sb = ttk.Scrollbar(logf, command=self.log.yview)
        sb.pack(side="right", fill="y", pady=8)
        self.log["yscrollcommand"] = sb.set

        self.write("Готово. 1) Установите зависимости (один раз). 2) Укажите папку локации. 3) Создать карту.\n")
        self.root.after(100, self.drain)

    # ---------- helpers ----------
    def write(self, text):
        self.q.put(text)

    def drain(self):
        try:
            while True:
                self.log.insert("end", self.q.get_nowait())
                self.log.see("end")
        except queue.Empty:
            pass
        self.root.after(100, self.drain)

    def set_busy(self, busy):
        self.busy = busy
        state = "disabled" if busy else "normal"
        for b in (self.btn_deps, self.btn_run):
            b["state"] = state

    def browse(self):
        d = filedialog.askdirectory(title="Выберите папку локации (внутри map_cache/5.0)")
        if d:
            self.path_var.set(d)

    def open_out(self):
        os.makedirs(self.outdir, exist_ok=True)
        try:
            os.startfile(self.outdir)  # Windows
        except AttributeError:
            subprocess.Popen(["xdg-open", self.outdir])

    def run_proc(self, args, done_msg):
        def worker():
            try:
                p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, encoding="utf-8", errors="replace", cwd=HERE)
                for line in p.stdout:
                    self.write(line)
                p.wait()
                self.write(f"\n{done_msg} (код {p.returncode})\n")
            except Exception as e:
                self.write(f"\nОшибка: {e}\n")
            finally:
                self.root.after(0, lambda: self.set_busy(False))
        self.set_busy(True)
        threading.Thread(target=worker, daemon=True).start()

    # ---------- actions ----------
    def install_deps(self):
        if self.busy:
            return
        self.write("\n>>> Установка зависимостей…\n")
        self.run_proc([sys.executable, "-m", "pip", "install", "-r", REQS], "Зависимости установлены.")

    def run_render(self):
        if self.busy:
            return
        path = self.path_var.get().strip()
        if not path or not os.path.isdir(path):
            self.write("\n! Сначала укажите существующую папку локации.\n")
            return
        self.write(f"\n>>> Рендер: {os.path.basename(path)}  [режим {self.mode.get()}, x{self.scale.get()}]\n")
        self.write("    (это может занять до ~1–2 минут на большую локацию)\n")
        self.run_proc([sys.executable, SCMAP, path, "-o", self.outdir,
                       "-m", self.mode.get(), "-s", str(self.scale.get())],
                      "Карта готова. Нажмите «Открыть результаты».")


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
