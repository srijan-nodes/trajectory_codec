"""
layer_decomp.py — Intrinsic Layer Decomposition for TrajCDDec V6
================================================================
Decouples geometric motion surfaces (diffuse base + contours)
from specular reflection fields (highlights, Fresnel glare).
"""
import cv2
import numpy as np

def decompose_frame(frame: np.ndarray, bg_threshold: int = 240, specular_threshold: int = 175):
    """
    Decomposes a grayscale frame into:
      1. Foreground / background mask
      2. Base motion surface (diffuse shading, geometry)
      3. Specular reflection field (high-intensity luminous highlight)
    """
    gray = frame.squeeze()
    
    # In ball.mp4, background is bright white (> 240)
    # Detect if background is light or dark
    corners = np.array([gray[0, 0], gray[0, -1], gray[-1, 0], gray[-1, -1]])
    is_light_bg = bool(np.mean(corners) > 128)
    
    if is_light_bg:
        fg_mask = gray < bg_threshold
        specular = np.zeros_like(gray, dtype=np.float32)
        specular[fg_mask] = np.maximum(0.0, gray[fg_mask].astype(np.float32) - specular_threshold)
        
        base_surface = gray.astype(np.float32).copy()
        base_surface[fg_mask] = np.minimum(gray[fg_mask].astype(np.float32), float(specular_threshold))
    else:
        fg_mask = gray > (255 - bg_threshold)
        specular = np.zeros_like(gray, dtype=np.float32)
        specular[fg_mask] = np.maximum(0.0, gray[fg_mask].astype(np.float32) - specular_threshold)
        
        base_surface = gray.astype(np.float32).copy()
        base_surface[fg_mask] = np.minimum(gray[fg_mask].astype(np.float32), float(specular_threshold))
        
    return {
        'fg_mask': fg_mask,
        'base_surface': base_surface,
        'specular': specular,
        'is_light_bg': is_light_bg
    }

def estimate_surface_and_specular_motion(prev_decomp: dict, curr_decomp: dict):
    """
    Estimates decoupled motion vectors:
      v_surf: bulk trajectory of the geometric motion surface (dy_s, dx_s)
      v_spec: optical trajectory of the specular reflection (dy_r, dx_r)
    """
    m_prev = prev_decomp['fg_mask']
    m_curr = curr_decomp['fg_mask']
    
    if not np.any(m_prev) or not np.any(m_curr):
        return (0, 0), (0, 0)
        
    # Surface centroid tracking
    cy_p, cx_p = np.mean(np.where(m_prev)[0]), np.mean(np.where(m_prev)[1])
    cy_c, cx_c = np.mean(np.where(m_curr)[0]), np.mean(np.where(m_curr)[1])
    dy_s = int(round(cy_p - cy_c))
    dx_s = int(round(cx_p - cx_c))
    
    # Specular centroid tracking
    s_prev = prev_decomp['specular'] > 0
    s_curr = curr_decomp['specular'] > 0
    
    if np.any(s_prev) and np.any(s_curr):
        sy_p, sx_p = np.mean(np.where(s_prev)[0]), np.mean(np.where(s_prev)[1])
        sy_c, sx_c = np.mean(np.where(s_curr)[0]), np.mean(np.where(s_curr)[1])
        dy_r = int(round(sy_p - sy_c))
        dx_r = int(round(sx_p - sx_c))
    else:
        dy_r, dx_r = dy_s, dx_s
        
    return (dy_s, dx_s), (dy_r, dx_r)

def composite_layers(surface: np.ndarray, specular: np.ndarray):
    """
    Additively composites base motion surface and specular reflection field.
    """
    recon = surface + specular
    return np.clip(recon, 0, 255).astype(np.uint8)
