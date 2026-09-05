"""
main.py -- Hybrid Video Codec GUI
====================================
Two tabs:

  Manual  -- Encode a video to .nam  OR  Decode a .nam to .mp4
  Batch   -- Run all videos in a folder, generate colour-coded XLSX

Save-video modes (radio):
  - None              : don't save any reconstructed video
  - Output only       : save decoded reconstruction as .mp4
  - Side-by-side      : stitch original + decoded into one comparison .mp4

All outputs land in a configurable output/ folder.
"""

import os
import sys
import glob
import threading
import tempfile
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from tkinterdnd2 import DND_FILES, TkinterDnD

import cv2
import numpy as np
import openpyxl
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
from openpyxl.utils import get_column_letter

sys.path.insert(0, str(Path(__file__).parent))
from encoder import encode
from decoder import decode


# ─────────────────────────────────────────────────────────────────────────────
# THEME
# ─────────────────────────────────────────────────────────────────────────────
BG      = "#0f1117"
BG2     = "#1a1d27"
BG3     = "#242736"
ACCENT  = "#6c63ff"
ACCENT2 = "#a78bfa"
SUCCESS = "#22c55e"
WARNING = "#f59e0b"
DANGER  = "#ef4444"
TEXT    = "#f1f5f9"
TEXTDIM = "#94a3b8"
BORDER  = "#2e3347"

FH1   = ("Segoe UI", 18, "bold")
FH2   = ("Segoe UI", 12, "bold")
FBODY = ("Segoe UI", 10)
FMONO = ("Consolas",  9)
FBTN  = ("Segoe UI", 10, "bold")

DEFAULT_OUT = str(Path(__file__).parent / "output")


# ─────────────────────────────────────────────────────────────────────────────
# VIDEO UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


def save_output_video(frames, out_path, original_path=None, fps=30):
    """Save decoded frames as mp4, restoring color if original is provided."""
    if not frames:
        return
    h, w = frames[0].shape[:2]
    cap = cv2.VideoCapture(original_path) if original_path else None
    if cap:
        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        if orig_fps and orig_fps > 0:
            fps = orig_fps
    vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h), isColor=True)
    
    for dec_f in frames:
        if dec_f.ndim == 3 and dec_f.shape[2] == 3:
            bgr = np.clip(dec_f, 0, 255).astype(np.uint8)
        elif cap:
            ret, orig_f = cap.read()
            if ret:
                orig_ycrcb = cv2.cvtColor(orig_f[:h, :w], cv2.COLOR_BGR2YCrCb)
                dec_ycrcb = orig_ycrcb.copy()
                dec_ycrcb[:,:,0] = np.clip(dec_f.squeeze(), 0, 255).astype(np.uint8)
                bgr = cv2.cvtColor(dec_ycrcb, cv2.COLOR_YCrCb2BGR)
            else:
                bgr = cv2.cvtColor(np.clip(dec_f.squeeze(), 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        else:
            bgr = cv2.cvtColor(np.clip(dec_f.squeeze(), 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        vw.write(bgr)
        
    if cap:
        cap.release()
    vw.release()


def save_stitched_video(frames, original_path, out_path, fps=30):
    """Save original | decoded side-by-side comparison mp4."""
    if not frames:
        return
    cap = cv2.VideoCapture(original_path)
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    if orig_fps and orig_fps > 0:
        fps = orig_fps
    h, w = frames[0].shape[:2]
    out_w = w * 2 + 4          # 4px divider
    vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, h), isColor=True)

    for dec_f in frames:
        ret, orig_f = cap.read()
        if not ret:
            break
        orig_bgr = orig_f[:h, :w].copy()
        
        if dec_f.ndim == 3 and dec_f.shape[2] == 3:
            dec_bgr = np.clip(dec_f, 0, 255).astype(np.uint8)
        else:
            # Colorize decoded Luma (dec_f) using original Chroma (U/V)
            orig_ycrcb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2YCrCb)
            dec_ycrcb = orig_ycrcb.copy()
            dec_ycrcb[:,:,0] = np.clip(dec_f.squeeze(), 0, 255).astype(np.uint8)
            dec_bgr = cv2.cvtColor(dec_ycrcb, cv2.COLOR_YCrCb2BGR)

        div = np.full((h, 4, 3), 80, dtype=np.uint8)
        row = np.concatenate([orig_bgr, div, dec_bgr], axis=1)

        # Labels
        cv2.putText(row, "ORIGINAL",  (8, 22),     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1, cv2.LINE_AA)
        cv2.putText(row, "DECODED",   (w+12, 22),   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120,210,120), 1, cv2.LINE_AA)
        vw.write(row)

    cap.release()
    vw.release()

def save_overlay_video(frames, mode_maps, original_path, out_path, fps=30):
    """Save diagnostic mode map overlay."""
    if not frames or not mode_maps:
        return
    cap = cv2.VideoCapture(original_path)
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    if orig_fps and orig_fps > 0:
        fps = orig_fps
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h), isColor=True)
    BLOCK_SIZE = 16
    colors = {
        255: (0, 255, 0),      7:   (0, 0, 255),      12:  (0, 255, 255),
        15:  (0, 128, 255),    9:   (255, 0, 255),    10:  (255, 0, 128),
        4:   (255, 255, 0),    11:  (255, 128, 0),    13:  (128, 255, 0),
        1:   (128, 128, 128),  2:   (255, 255, 255),  6:   (0, 100, 0),
    }
    for i, dec_f in enumerate(frames):
        ret, orig_f = cap.read()
        if not ret: break
        orig_bgr = orig_f[:h, :w].copy()
        if dec_f.ndim == 3 and dec_f.shape[2] == 3:
            dec_bgr = np.clip(dec_f, 0, 255).astype(np.uint8)
        else:
            orig_ycrcb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2YCrCb)
            dec_ycrcb = orig_ycrcb.copy()
            dec_ycrcb[:,:,0] = np.clip(dec_f.squeeze(), 0, 255).astype(np.uint8)
            dec_bgr = cv2.cvtColor(dec_ycrcb, cv2.COLOR_YCrCb2BGR)
        overlay = dec_bgr.copy()
        m_map = mode_maps[i]
        for y in range(m_map.shape[0]):
            for x in range(m_map.shape[1]):
                c = colors.get(m_map[y, x], (50, 50, 50))
                overlay[y*BLOCK_SIZE:(y+1)*BLOCK_SIZE, x*BLOCK_SIZE:(x+1)*BLOCK_SIZE] = c
        blended = cv2.addWeighted(dec_bgr, 0.7, overlay, 0.3, 0)
        vw.write(blended)
    vw.release()
    cap.release()


# ─────────────────────────────────────────────────────────────────────────────
# XLSX WRITER
# ─────────────────────────────────────────────────────────────────────────────
def _rfill(r):
    if r is None: return PatternFill(fill_type=None)
    if r <= 0.50: return PatternFill("solid", fgColor="00B050")
    if r <= 0.70: return PatternFill("solid", fgColor="92D050")
    if r <= 0.80: return PatternFill("solid", fgColor="FFFF00")
    if r <= 1.00: return PatternFill("solid", fgColor="FFC000")
    return PatternFill("solid", fgColor="FF0000")

def _mfill(m):
    if m is None: return PatternFill(fill_type=None)
    if m <= 10:   return PatternFill("solid", fgColor="00B050")
    if m <= 50:   return PatternFill("solid", fgColor="92D050")
    if m <= 150:  return PatternFill("solid", fgColor="FFFF00")
    if m <= 400:  return PatternFill("solid", fgColor="FFC000")
    return PatternFill("solid", fgColor="FF0000")

def _sfill(s):
    if s is None: return PatternFill(fill_type=None)
    if s >= 10:   return PatternFill("solid", fgColor="00B050")
    if s >= 5:    return PatternFill("solid", fgColor="92D050")
    if s >= 2:    return PatternFill("solid", fgColor="FFFF00")
    return PatternFill("solid", fgColor="FFC000")

XCOLS = [
    ("Video", 22), ("Orig KB", 10), ("Comp KB", 10), ("Ratio", 10),
    ("<0.8x", 8),  ("Avg MSE", 10), ("Quality", 18), ("Enc s", 8),
    ("Enc FPS", 9),("Dec s", 8),    ("V5 fr", 8),    ("V6 fr", 8),
    ("Total", 8),  ("V6 %", 7),
]

def write_xlsx(results, out_path):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Hybrid Codec Report"
    hf = PatternFill("solid", fgColor="1F3864")
    hfont = Font(bold=True, color="FFFFFF", size=11)
    alt   = PatternFill("solid", fgColor="EBF3FB")
    bdr   = Border(
        left  =Side(style="thin", color="BBBBBB"),
        right =Side(style="thin", color="BBBBBB"),
        top   =Side(style="thin", color="BBBBBB"),
        bottom=Side(style="thin", color="BBBBBB"),
    )

    ws.merge_cells("A1:N1")
    c = ws["A1"]
    c.value = "Dynamic Hybrid Video Codec -- Batch Test Report"
    c.font  = Font(bold=True, size=14, color="FFFFFF")
    c.fill  = hf
    c.alignment = Alignment(horizontal="center", vertical="center")
    ws.row_dimensions[1].height = 30

    for ci, (name, w) in enumerate(XCOLS, 1):
        cell = ws.cell(row=2, column=ci, value=name)
        cell.fill = hf; cell.font = hfont; cell.border = bdr
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        ws.column_dimensions[get_column_letter(ci)].width = w
    ws.row_dimensions[2].height = 36

    for ri, r in enumerate(results, 3):
        enc, dec = r.get("enc"), r.get("dec")
        orig = enc["orig_bytes"]       if enc else None
        comp = enc["compressed_bytes"] if enc else None
        rat  = enc["ratio"]            if enc else None
        et   = enc["encode_time_s"]    if enc else None
        v5   = enc.get("v5_frames", None)        if enc else None
        v6   = enc.get("v6_frames", enc.get("total_frames", None)) if enc else None
        tot  = enc.get("total_frames", None)     if enc else None
        vp   = round(100*v6/tot,1)     if (v6 is not None and tot) else None
        mse  = dec["avg_mse"]          if dec else None
        dt   = dec["dec_time"]         if dec else None
        efps = r.get("enc_fps")
        ok   = "YES" if (rat is not None and rat < 0.80) else "NO"
        qual = ("Visually Lossless" if mse and mse<=10 else "Good" if mse and mse<=50
                else "Acceptable" if mse and mse<=150 else "Degraded" if mse and mse<=400
                else "Poor" if mse else "N/A")

        row = [
            r["name"],
            f"{orig/1024:.1f}" if orig else "N/A",
            f"{comp/1024:.1f}" if comp else "N/A",
            f"{rat:.3f}x"      if rat is not None else "N/A",
            ok, round(mse,2) if mse else "N/A", qual,
            et if et else "N/A", round(efps,1) if efps else "N/A",
            dt if dt else "N/A", v5 or "N/A", v6 or "N/A",
            tot or "N/A", f"{vp}%" if vp is not None else "N/A",
        ]
        af = alt if ri % 2 == 0 else None
        for ci, val in enumerate(row, 1):
            cell = ws.cell(row=ri, column=ci, value=val)
            cell.alignment = Alignment(horizontal="center", vertical="center")
            cell.border = bdr
            if af: cell.fill = af

        ws.cell(row=ri, column=4).fill = _rfill(rat)
        oc = ws.cell(row=ri, column=5)
        oc.fill = PatternFill("solid", fgColor="00B050" if ok=="YES" else "FF0000")
        oc.font = Font(bold=True, color="FFFFFF")
        ws.cell(row=ri, column=6).fill = _mfill(mse)
        ws.cell(row=ri, column=9).fill = _sfill(efps)
        ws.row_dimensions[ri].height = 20

    ws.freeze_panes = "A3"
    wb.save(out_path)


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: quality label
# ─────────────────────────────────────────────────────────────────────────────
def _qual(mse):
    if mse is None:   return "N/A",    TEXTDIM
    if mse <= 10:     return "Visually Lossless", SUCCESS
    if mse <= 50:     return "Good",   SUCCESS
    if mse <= 150:    return "Acceptable", WARNING
    if mse <= 400:    return "Degraded", WARNING
    return "Poor", DANGER


# ─────────────────────────────────────────────────────────────────────────────
# APPLICATION
# ─────────────────────────────────────────────────────────────────────────────
class App(TkinterDnD.Tk):
    def __init__(self):
        super().__init__()
        self.title("Hybrid Video Codec")
        self.geometry("1060x700")
        self.minsize(860, 580)
        self.configure(bg=BG)
        self._style()
        self._header()
        self._notebook()

    # ── Styles ───────────────────────────────────────────────────────────────
    def _style(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure(".", background=BG, foreground=TEXT, font=FBODY)

        s.configure("TNotebook", background=BG, borderwidth=0)
        s.configure("TNotebook.Tab", background=BG3, foreground=TEXTDIM,
                     padding=[20, 9], font=FH2, borderwidth=0)
        s.map("TNotebook.Tab",
              background=[("selected", ACCENT)],
              foreground=[("selected", TEXT)])

        s.configure("A.TButton", background=ACCENT, foreground=TEXT,
                     font=FBTN, borderwidth=0, padding=[18, 9])
        s.map("A.TButton", background=[("active", ACCENT2), ("disabled", BG3)],
              foreground=[("disabled", TEXTDIM)])

        s.configure("G.TButton", background=BG3, foreground=TEXTDIM,
                     font=FBTN, borderwidth=0, padding=[14, 9])
        s.map("G.TButton", background=[("active", BORDER)],
              foreground=[("active", TEXT)])

        s.configure("TEntry", fieldbackground=BG3, foreground=TEXT,
                     insertcolor=TEXT, borderwidth=0, relief="flat", padding=6)

        s.configure("Bar.Horizontal.TProgressbar",
                     background=ACCENT, troughcolor=BG3, borderwidth=0, thickness=7)

        s.configure("TRadiobutton", background=BG2, foreground=TEXT,
                     indicatorcolor=ACCENT, font=FBODY)
        s.map("TRadiobutton", background=[("active", BG2)],
              indicatorcolor=[("selected", ACCENT)])

        s.configure("Card.TFrame", background=BG2)
        s.configure("TFrame", background=BG)
        s.configure("TLabel", background=BG, foreground=TEXT, font=FBODY)
        s.configure("D.TLabel", background=BG2, foreground=TEXTDIM, font=FBODY)
        s.configure("H.TLabel", background=BG2, foreground=ACCENT2, font=FH2)
        s.configure("Mono.TLabel", background=BG2, foreground=TEXT, font=FMONO)

        s.configure("Res.Treeview", background=BG2, foreground=TEXT,
                     fieldbackground=BG2, rowheight=24, font=FBODY, borderwidth=0)
        s.configure("Res.Treeview.Heading", background=BG3, foreground=ACCENT2,
                     font=("Segoe UI", 9, "bold"), borderwidth=0)
        s.map("Res.Treeview", background=[("selected", ACCENT)],
              foreground=[("selected", TEXT)])

    # ── Header bar ───────────────────────────────────────────────────────────
    def _header(self):
        h = tk.Frame(self, bg=BG2, height=60)
        h.pack(fill="x")
        h.pack_propagate(False)
        tk.Label(h, text="  HYBRID VIDEO CODEC",
                 font=FH1, bg=BG2, fg=TEXT).pack(side="left", padx=16, pady=10)
        tk.Label(h, text="V5 Macroblock  +  V6 Vector  |  Dynamic Optical-Flow Routing",
                 font=FBODY, bg=BG2, fg=TEXTDIM).pack(side="left", padx=4)
        tk.Label(h, text="  v1.0  ", font=("Segoe UI", 9, "bold"),
                 bg=ACCENT, fg=TEXT).pack(side="right", padx=16, pady=16)

    # ── Notebook tabs ─────────────────────────────────────────────────────────
    def _notebook(self):
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True)
        m = ttk.Frame(nb); b = ttk.Frame(nb)
        nb.add(m, text="  Manual  ")
        nb.add(b, text="  Batch Test Runner  ")
        self._manual(m)
        self._batch(b)

    # ─────────────────────────────────────────────────────────────────────────
    # DND HELPER
    # ─────────────────────────────────────────────────────────────────────────
    def _bind_dnd(self, widget, var, is_dir=False):
        widget.drop_target_register(DND_FILES)
        def on_drop(event):
            path = event.data
            if path.startswith('{') and path.endswith('}'):
                path = path[1:-1]
            if is_dir and os.path.isfile(path):
                path = os.path.dirname(path)
            var.set(path)
        widget.dnd_bind('<<Drop>>', on_drop)

    # ─────────────────────────────────────────────────────────────────────────
    # MANUAL TAB
    # ─────────────────────────────────────────────────────────────────────────
    def _manual(self, p):

        # ── Output folder ─────────────────────────────────────────────────────
        of = ttk.Frame(p, style="Card.TFrame")
        of.pack(fill="x", padx=20, pady=(18, 0))
        ttk.Label(of, text="Output Folder (Drag & Drop allowed)", style="H.TLabel").grid(
            row=0, column=0, sticky="w", padx=14, pady=(12,3))
        self.m_outdir = tk.StringVar(value=DEFAULT_OUT)
        out_entry = ttk.Entry(of, textvariable=self.m_outdir, font=FMONO, width=60)
        out_entry.grid(row=1, column=0, sticky="ew", padx=14, pady=(0,12))
        self._bind_dnd(out_entry, self.m_outdir, is_dir=True)
        ttk.Button(of, text="Browse...", style="G.TButton",
                   command=lambda: self._pick_dir(self.m_outdir)).grid(
            row=1, column=1, padx=(6,14), pady=(0,12))
        of.columnconfigure(0, weight=1)

        # ── Two operation cards side by side ─────────────────────────────────
        ops = ttk.Frame(p)
        ops.pack(fill="x", padx=20, pady=(10, 0))
        ops.columnconfigure(0, weight=1, uniform="op")
        ops.columnconfigure(1, weight=1, uniform="op")

        # ENCODE card
        ec = ttk.Frame(ops, style="Card.TFrame")
        ec.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        ttk.Label(ec, text="Encode  ( video -> .nam )  [Drag & Drop]", style="H.TLabel").pack(
            anchor="w", padx=14, pady=(12, 6))

        self.m_src = tk.StringVar()
        src_row = ttk.Frame(ec, style="Card.TFrame")
        src_row.pack(fill="x", padx=14, pady=(0, 10))
        src_entry = ttk.Entry(src_row, textvariable=self.m_src, font=FMONO)
        src_entry.pack(side="left", fill="x", expand=True)
        self._bind_dnd(src_entry, self.m_src)
        
        # When dragged in, auto-fill output fields
        def _on_src_change(*args):
            p = self.m_src.get()
            if p and os.path.isfile(p):
                self.m_orig.set(p) # pre-fill original
        self.m_src.trace_add("write", _on_src_change)

        ttk.Button(src_row, text="Browse", style="G.TButton",
                   command=self._m_browse_src).pack(side="left", padx=(6, 0))

        self.m_enc_btn = ttk.Button(ec, text="  Encode  ",
                                    style="A.TButton", command=self._m_encode)
        self.m_enc_btn.pack(padx=14, pady=(0, 10), anchor="w")

        self.m_enc_status = tk.Label(ec, text="", bg=BG2, fg=TEXTDIM, font=FBODY)
        self.m_enc_status.pack(anchor="w", padx=14, pady=(0, 8))

        self.m_enc_pbar = ttk.Progressbar(ec, orient="horizontal",
                                           mode="indeterminate",
                                           style="Bar.Horizontal.TProgressbar")
        self.m_enc_pbar.pack(fill="x", padx=14, pady=(0, 14))

        # DECODE card
        dc = ttk.Frame(ops, style="Card.TFrame")
        dc.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        ttk.Label(dc, text="Decode  ( .nam -> .mp4 )  [Drag & Drop]", style="H.TLabel").pack(
            anchor="w", padx=14, pady=(12, 6))

        self.m_nam = tk.StringVar()
        nam_row = ttk.Frame(dc, style="Card.TFrame")
        nam_row.pack(fill="x", padx=14, pady=(0, 6))
        nam_entry = ttk.Entry(nam_row, textvariable=self.m_nam, font=FMONO)
        nam_entry.pack(side="left", fill="x", expand=True)
        self._bind_dnd(nam_entry, self.m_nam)
        ttk.Button(nam_row, text="Browse", style="G.TButton",
                   command=self._m_browse_nam).pack(side="left", padx=(6, 0))

        # Save mode radio
        ttk.Label(dc, text="Save video as:", style="D.TLabel").pack(
            anchor="w", padx=14, pady=(2, 2))
        self.m_save_mode = tk.StringVar(value="none")
        save_frame = ttk.Frame(dc, style="Card.TFrame")
        save_frame.pack(anchor="w", padx=14, pady=(0, 6))
        for val, label in [("none", "Don't save"),
                            ("output", "Output only"),
                            ("stitch", "Side-by-side with original"),
                            ("overlay", "Diagnostic Overlay")]:
            ttk.Radiobutton(save_frame, text=label, variable=self.m_save_mode,
                            value=val, style="TRadiobutton").pack(
                side="left", padx=(0, 10))

        # Original video field (for stitch mode)
        self.m_orig_frame = ttk.Frame(dc, style="Card.TFrame")
        self.m_orig_frame.pack(fill="x", padx=14, pady=(0, 6))
        ttk.Label(self.m_orig_frame, text="Original video (for stitch):",
                  style="D.TLabel").pack(anchor="w")
        self.m_orig = tk.StringVar()
        orig_row = ttk.Frame(self.m_orig_frame, style="Card.TFrame")
        orig_row.pack(fill="x")
        orig_entry = ttk.Entry(orig_row, textvariable=self.m_orig, font=FMONO)
        orig_entry.pack(side="left", fill="x", expand=True)
        self._bind_dnd(orig_entry, self.m_orig)
        ttk.Button(orig_row, text="Browse", style="G.TButton",
                   command=self._m_browse_orig).pack(side="left", padx=(6, 0))

        self.m_save_mode.trace_add("write", self._m_toggle_orig)
        self._m_toggle_orig()

        self.m_dec_btn = ttk.Button(dc, text="  Decode  ",
                                    style="A.TButton", command=self._m_decode)
        self.m_dec_btn.pack(padx=14, pady=(2, 10), anchor="w")

        self.m_dec_status = tk.Label(dc, text="", bg=BG2, fg=TEXTDIM, font=FBODY)
        self.m_dec_status.pack(anchor="w", padx=14, pady=(0, 8))

        self.m_dec_pbar = ttk.Progressbar(dc, orient="horizontal",
                                           mode="indeterminate",
                                           style="Bar.Horizontal.TProgressbar")
        self.m_dec_pbar.pack(fill="x", padx=14, pady=(0, 14))

        # ── Stats panel ──────────────────────────────────────────────────────
        sc = ttk.Frame(p, style="Card.TFrame")
        sc.pack(fill="both", expand=True, padx=20, pady=(10, 16))
        ttk.Label(sc, text="Stats", style="H.TLabel").grid(
            row=0, column=0, columnspan=8, sticky="w", padx=14, pady=(10, 6))

        keys = [
            ("Orig Size", "orig"),   ("Comp Size", "comp"),
            ("Ratio",     "ratio"),  ("< 0.8x",    "ok"),
            ("Avg MSE",   "mse"),    ("Quality",    "qual"),
            ("Enc Time",  "etime"),  ("Dec Time",   "dtime"),
            ("Enc FPS",   "efps"),   ("V5 Frames",  "v5"),
            ("V6 Frames", "v6"),     ("Total",      "tot"),
        ]
        self._stat = {}
        for i, (label, key) in enumerate(keys):
            c = (i % 4) * 2
            r = i // 4 + 1
            tk.Label(sc, text=label + ":", bg=BG2, fg=TEXTDIM,
                     font=("Segoe UI", 9)).grid(row=r, column=c, sticky="e",
                                                 padx=(12, 3), pady=3)
            lbl = tk.Label(sc, text="--", bg=BG2, fg=TEXT, font=FMONO, anchor="w", width=18)
            lbl.grid(row=r, column=c+1, sticky="w", padx=(0, 10), pady=3)
            self._stat[key] = lbl

    def _m_toggle_orig(self, *_):
        vis = self.m_save_mode.get() == "stitch"
        if vis:
            self.m_orig_frame.pack(fill="x", padx=14, pady=(0, 6),
                                   before=self.m_dec_btn)
        else:
            self.m_orig_frame.pack_forget()

    def _pick_dir(self, var):
        p = filedialog.askdirectory(title="Select output folder")
        if p: var.set(p)

    def _m_browse_src(self):
        p = filedialog.askopenfilename(
            title="Select source video",
            filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv"), ("All", "*.*")])
        if p:
            self.m_src.set(p)
            self.m_orig.set(p)   # pre-fill for stitch mode too

    def _m_browse_nam(self):
        p = filedialog.askopenfilename(
            title="Select .nam file",
            filetypes=[("NAM", "*.nam"), ("All", "*.*")])
        if p: self.m_nam.set(p)

    def _m_browse_orig(self):
        p = filedialog.askopenfilename(
            title="Select original video (for stitch)",
            filetypes=[("Video", "*.mp4 *.avi"), ("All", "*.*")])
        if p: self.m_orig.set(p)

    # Encode
    def _m_encode(self):
        src = self.m_src.get().strip()
        if not src or not os.path.isfile(src):
            messagebox.showerror("Error", "Select a valid source video first.")
            return
        outdir = _ensure_dir(self.m_outdir.get().strip() or DEFAULT_OUT)
        nam    = os.path.join(outdir, Path(src).stem + ".nam")
        self.m_enc_btn.configure(state="disabled")
        self.m_enc_pbar.start(10)
        self.m_enc_status.configure(text="Encoding...", fg=TEXTDIM)

        def _work():
            try:
                enc = encode(src, nam)
                efps = enc["total_frames"] / enc["encode_time_s"] if enc.get("encode_time_s") else 0
                self.after(0, lambda: self._m_enc_done(enc, efps, nam))
            except Exception as e:
                self.after(0, lambda: self._m_enc_err(str(e)))

        threading.Thread(target=_work, daemon=True).start()

    def _m_enc_done(self, enc, efps, nam):
        self.m_enc_pbar.stop()
        self.m_enc_btn.configure(state="normal")
        self.m_enc_status.configure(text=f"Done -> {Path(nam).name}", fg=SUCCESS)
        self.m_nam.set(nam)    # auto-fill the decode field
        r = enc.get("ratio", 0)
        self._stat["orig"].configure(text=f"{enc['orig_bytes']/1024:.1f} KB")
        self._stat["comp"].configure(text=f"{enc['compressed_bytes']/1024:.1f} KB")
        self._stat["ratio"].configure(text=f"{r:.3f}x",
                                       fg=SUCCESS if r < 0.80 else DANGER)
        self._stat["ok"].configure(text="YES" if r < 0.80 else "NO",
                                    fg=SUCCESS if r < 0.80 else DANGER)
        self._stat["etime"].configure(text=f"{enc['encode_time_s']:.1f} s")
        self._stat["efps"].configure(text=f"{efps:.1f} fps")
        self._stat["v5"].configure(text=str(enc.get("v5_frames", "?")))
        self._stat["v6"].configure(text=str(enc.get("v6_frames", "?")))
        self._stat["tot"].configure(text=str(enc.get("total_frames", "?")))

    def _m_enc_err(self, msg):
        self.m_enc_pbar.stop()
        self.m_enc_btn.configure(state="normal")
        self.m_enc_status.configure(text=f"Error: {msg}", fg=DANGER)

    # Decode
    def _m_decode(self):
        nam  = self.m_nam.get().strip()
        if not nam or not os.path.isfile(nam):
            messagebox.showerror("Error", "Select a valid .nam file first.")
            return
        mode   = self.m_save_mode.get()
        orig   = self.m_orig.get().strip()
        if mode == "stitch" and (not orig or not os.path.isfile(orig)):
            messagebox.showerror("Error", "Select the original video for side-by-side mode.")
            return
        outdir = _ensure_dir(self.m_outdir.get().strip() or DEFAULT_OUT)
        self.m_dec_btn.configure(state="disabled")
        self.m_dec_pbar.start(10)
        self.m_dec_status.configure(text="Decoding...", fg=TEXTDIM)

        def _work():
            try:
                dec = decode(nam, original_video=orig if orig else None, return_mode_map=(mode=="overlay"))
                # Save video
                stem = Path(nam).stem
                if mode == "output":
                    vpath = os.path.join(outdir, stem + "_decoded.mp4")
                    save_output_video(dec["frames"], vpath, original_path=orig if orig else None,
                                      fps=dec.get("fps", 30))
                elif mode == "stitch":
                    vpath = os.path.join(outdir, stem + "_comparison.mp4")
                    save_stitched_video(dec["frames"], orig, vpath,
                                        fps=dec.get("fps", 30))
                elif mode == "overlay":
                    vpath = os.path.join(outdir, stem + "_overlay.mp4")
                    save_overlay_video(dec["frames"], dec.get("mode_maps"), orig, vpath,
                                        fps=dec.get("fps", 30))
                else:
                    vpath = None
                self.after(0, lambda: self._m_dec_done(dec, vpath))
            except Exception as e:
                self.after(0, lambda: self._m_dec_err(str(e)))

        threading.Thread(target=_work, daemon=True).start()

    def _m_dec_done(self, dec, vpath):
        self.m_dec_pbar.stop()
        self.m_dec_btn.configure(state="normal")
        msg = f"Done -> {Path(vpath).name}" if vpath else "Done (no video saved)"
        self.m_dec_status.configure(text=msg, fg=SUCCESS)
        mse = dec.get("avg_mse")
        ql, qc = _qual(mse)
        self._stat["mse"].configure(text=f"{mse:.2f}" if mse else "--")
        self._stat["qual"].configure(text=ql, fg=qc)
        self._stat["dtime"].configure(text=f"{dec['decode_time_s']:.1f} s")

    def _m_dec_err(self, msg):
        self.m_dec_pbar.stop()
        self.m_dec_btn.configure(state="normal")
        self.m_dec_status.configure(text=f"Error: {msg}", fg=DANGER)

    # ─────────────────────────────────────────────────────────────────────────
    # BATCH TAB
    # ─────────────────────────────────────────────────────────────────────────
    def _batch(self, p):

        # ── Folder + output dir ───────────────────────────────────────────────
        top = ttk.Frame(p, style="Card.TFrame")
        top.pack(fill="x", padx=20, pady=(18, 0))

        ttk.Label(top, text="Test Video Folder (Drag & Drop allowed)", style="H.TLabel").grid(
            row=0, column=0, sticky="w", padx=14, pady=(12, 3))
        self.b_folder = tk.StringVar(
            value=str(Path(__file__).parent.parent / "test_vid"))
        fold_entry = ttk.Entry(top, textvariable=self.b_folder, font=FMONO, width=52)
        fold_entry.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 4))
        self._bind_dnd(fold_entry, self.b_folder, is_dir=True)
        ttk.Button(top, text="Browse...", style="G.TButton",
                   command=lambda: self._pick_dir(self.b_folder)).grid(
            row=1, column=1, padx=(6, 14), pady=(0, 4))

        ttk.Label(top, text="Output Folder (Drag & Drop allowed)", style="H.TLabel").grid(
            row=2, column=0, sticky="w", padx=14, pady=(6, 3))
        self.b_outdir = tk.StringVar(value=DEFAULT_OUT)
        out_entry = ttk.Entry(top, textvariable=self.b_outdir, font=FMONO, width=52)
        out_entry.grid(row=3, column=0, sticky="ew", padx=14, pady=(0, 12))
        self._bind_dnd(out_entry, self.b_outdir, is_dir=True)
        ttk.Button(top, text="Browse...", style="G.TButton",
                   command=lambda: self._pick_dir(self.b_outdir)).grid(
            row=3, column=1, padx=(6, 14), pady=(0, 12))
        top.columnconfigure(0, weight=1)

        # ── Save mode ─────────────────────────────────────────────────────────
        sm = ttk.Frame(p, style="Card.TFrame")
        sm.pack(fill="x", padx=20, pady=(8, 0))
        ttk.Label(sm, text="Save decoded videos as:", style="D.TLabel").pack(
            side="left", padx=14, pady=10)
        self.b_save_mode = tk.StringVar(value="stitch")
        for val, label in [("none",   "Don't save"),
                            ("output", "Output only"),
                            ("stitch", "Side-by-side with original"),
                            ("overlay", "Diagnostic Overlay")]:
            ttk.Radiobutton(sm, text=label, variable=self.b_save_mode,
                            value=val, style="TRadiobutton").pack(
                side="left", padx=(0, 14), pady=10)

        # ── Run row ───────────────────────────────────────────────────────────
        rr = ttk.Frame(p)
        rr.pack(fill="x", padx=20, pady=(10, 0))
        self.b_run_btn = ttk.Button(rr, text="  Run Batch  ",
                                    style="A.TButton", command=self._b_run)
        self.b_run_btn.pack(side="left")
        self.b_xlsx_btn = ttk.Button(rr, text="  Open Report (.xlsx)  ",
                                     style="G.TButton", command=self._b_open_xlsx,
                                     state="disabled")
        self.b_xlsx_btn.pack(side="left", padx=8)
        self.b_out_btn = ttk.Button(rr, text="  Open Output Folder  ",
                                     style="G.TButton", command=self._b_open_folder)
        self.b_out_btn.pack(side="left", padx=(0, 8))
        self.b_status = tk.Label(rr, text="Ready.", bg=BG, fg=TEXTDIM, font=FBODY)
        self.b_status.pack(side="left", padx=10)

        self.b_pbar = ttk.Progressbar(p, orient="horizontal", mode="determinate",
                                       style="Bar.Horizontal.TProgressbar",
                                       maximum=100)
        self.b_pbar.pack(fill="x", padx=20, pady=(8, 0))

        # ── Results table ─────────────────────────────────────────────────────
        tf = ttk.Frame(p, style="Card.TFrame")
        tf.pack(fill="both", expand=True, padx=20, pady=(10, 16))
        ttk.Label(tf, text="Results", style="H.TLabel").pack(
            anchor="w", padx=14, pady=(10, 6))

        cols = ("Video", "Orig KB", "Comp KB", "Ratio", "<0.8x",
                "MSE", "Quality", "Enc FPS", "V5", "V6")
        self.b_tree = ttk.Treeview(tf, columns=cols, show="headings",
                                    style="Res.Treeview", height=12)
        widths = [170, 75, 75, 70, 55, 70, 145, 68, 50, 50]
        for c, w in zip(cols, widths):
            self.b_tree.heading(c, text=c)
            self.b_tree.column(c, width=w, anchor="center", minwidth=45)

        self.b_tree.tag_configure("green",  background="#0d2a1a", foreground=SUCCESS)
        self.b_tree.tag_configure("yellow", background="#2a2200", foreground=WARNING)
        self.b_tree.tag_configure("red",    background="#2a0d0d", foreground=DANGER)
        self.b_tree.tag_configure("alt",    background="#1e2130")

        vsb = ttk.Scrollbar(tf, orient="vertical", command=self.b_tree.yview)
        self.b_tree.configure(yscrollcommand=vsb.set)
        self.b_tree.pack(side="left", fill="both", expand=True, padx=(14, 0), pady=(0, 14))
        vsb.pack(side="right", fill="y", pady=(0, 14), padx=(0, 8))

        self._b_xlsx_path = None

    def _b_open_xlsx(self):
        if self._b_xlsx_path and os.path.isfile(self._b_xlsx_path):
            os.startfile(self._b_xlsx_path)

    def _b_open_folder(self):
        outdir = self.b_outdir.get().strip() or DEFAULT_OUT
        if os.path.isdir(outdir):
            os.startfile(outdir)

    def _b_run(self):
        folder = self.b_folder.get().strip()
        if not folder or not os.path.isdir(folder):
            messagebox.showerror("Error", "Select a valid video folder.")
            return
        videos = sorted(glob.glob(os.path.join(folder, "*.mp4")) +
                        glob.glob(os.path.join(folder, "*.avi")))
        if not videos:
            messagebox.showerror("Error", "No .mp4 / .avi files found.")
            return

        outdir = _ensure_dir(self.b_outdir.get().strip() or DEFAULT_OUT)
        mode   = self.b_save_mode.get()

        for row in self.b_tree.get_children():
            self.b_tree.delete(row)
        self.b_run_btn.configure(state="disabled")
        self.b_xlsx_btn.configure(state="disabled")
        self.b_pbar.configure(maximum=len(videos), value=0)

        def _work():
            results = []
            tmp = tempfile.mkdtemp(prefix="hybd_")
            for i, vpath in enumerate(videos):
                name  = Path(vpath).name
                npath = os.path.join(tmp, Path(vpath).stem + ".nam")
                self.after(0, lambda n=name, i=i: self.b_status.configure(
                    text=f"[{i+1}/{len(videos)}] Encoding {n}..."))

                enc, efps = None, None
                try:
                    enc  = encode(vpath, npath)
                    efps = enc["total_frames"] / enc["encode_time_s"] if enc.get("encode_time_s") else 0
                    # Copy .nam to output folder
                    import shutil
                    shutil.copy2(npath, os.path.join(outdir, Path(vpath).stem + ".nam"))
                except Exception as e:
                    print(f"Encode err {name}: {e}")

                dec = None
                if enc:
                    self.after(0, lambda n=name, i=i: self.b_status.configure(
                        text=f"[{i+1}/{len(videos)}] Decoding {n}..."))
                    try:
                        dec = decode(npath, original_video=vpath, return_mode_map=(mode=="overlay"))
                        stem = Path(vpath).stem
                        if mode == "output" and dec.get("frames"):
                            save_output_video(dec["frames"],
                                              os.path.join(outdir, stem+"_decoded.mp4"),
                                              original_path=vpath,
                                              fps=enc.get("fps", 30))
                        elif mode == "stitch" and dec.get("frames"):
                            save_stitched_video(dec["frames"], vpath,
                                                os.path.join(outdir, stem+"_comparison.mp4"),
                                                fps=enc.get("fps", 30))
                        elif mode == "overlay" and dec.get("frames") and dec.get("mode_maps"):
                            save_overlay_video(dec["frames"], dec.get("mode_maps"), vpath,
                                                os.path.join(outdir, stem+"_overlay.mp4"),
                                                fps=enc.get("fps", 30))
                    except Exception as e:
                        print(f"Decode err {name}: {e}")

                r = {"name": name, "enc": enc, "dec": dec, "enc_fps": efps}
                results.append(r)
                self.after(0, lambda rr=r, idx=i: self._b_row(rr, idx))
                self.after(0, lambda v=i+1: self.b_pbar.configure(value=v))

                try: os.remove(npath)
                except: pass

            try: os.rmdir(tmp)
            except: pass

            xlsx = os.path.join(outdir, "hybrid_batch_report.xlsx")
            write_xlsx(results, xlsx)
            self._b_xlsx_path = xlsx
            self.after(0, lambda: self.b_status.configure(
                text=f"Done! {len(videos)} videos. Report -> {Path(xlsx).name}", fg=SUCCESS))
            self.after(0, lambda: self.b_run_btn.configure(state="normal"))
            self.after(0, lambda: self.b_xlsx_btn.configure(state="normal"))

        threading.Thread(target=_work, daemon=True).start()

    def _b_row(self, r, idx):
        enc  = r.get("enc")
        dec  = r.get("dec")
        efps = r.get("enc_fps")
        rat  = enc["ratio"]       if enc else None
        mse  = dec["avg_mse"]     if dec else None
        ql, _ = _qual(mse)
        tag = ("green"  if rat is not None and rat <= 0.70 and (not mse or mse <= 50)
               else "yellow" if rat is not None and rat <= 0.80
               else "red")
        if idx % 2 == 0 and tag == "yellow":
            tag = "alt"
        self.b_tree.insert("", "end", values=(
            r["name"],
            f"{enc['orig_bytes']/1024:.1f}"       if enc else "ERR",
            f"{enc['compressed_bytes']/1024:.1f}" if enc else "ERR",
            f"{rat:.3f}x"                          if rat else "ERR",
            "YES" if (rat and rat<0.80) else "NO",
            f"{mse:.1f}"                           if mse else "ERR",
            ql,
            f"{efps:.1f}"                          if efps else "ERR",
            enc["v5_frames"] if enc else "?",
            enc["v6_frames"] if enc else "?",
        ), tags=(tag,))
        self.b_tree.yview_moveto(1.0)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    App().mainloop()
