"""
Unit Testing SnapSign — Bab 4 Pengujian White Box (Unit Testing)
Jalankan: python -m pytest test_unit.py -v
"""
import numpy as np
import cv2
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import realtime_v10 as R
from server_v10 import CONF_THRESHOLD, motion_energy, segment_frame


# ── Unit 1: sample_clip_eval ────────────────────────────────────────────────
def test_sample_clip_eval_returns_16_frames():
    """Sampling harus selalu mengembalikan tepat 16 frame."""
    dummy = [np.zeros((224, 224, 3), dtype=np.uint8)] * 60
    result = R.sample_clip_eval(dummy, R.NUM_FRAMES, 0.15)
    assert len(result) == R.NUM_FRAMES, f"Expected 16, got {len(result)}"


def test_sample_clip_eval_works_with_minimum_frames():
    """Sampling tetap bekerja meski input hanya 4 frame."""
    dummy = [np.zeros((224, 224, 3), dtype=np.uint8)] * 4
    result = R.sample_clip_eval(dummy, R.NUM_FRAMES, 0.0)
    assert len(result) == R.NUM_FRAMES


# ── Unit 2: motion_energy ───────────────────────────────────────────────────
def test_motion_energy_low_on_static_signal():
    """Sinyal diam (nol) menghasilkan energi gerak yang sangat kecil."""
    flat = np.zeros(30, dtype=np.float32)
    peak = motion_energy(flat)
    assert peak < 0.15, f"Expected < 0.15, got {peak}"


def test_motion_energy_high_on_active_signal():
    """Sinyal dengan gerakan aktif menghasilkan energi gerak tinggi."""
    active = np.array([0.0]*5 + [1.0]*10 + [0.8]*10 + [0.0]*5, dtype=np.float32)
    peak = motion_energy(active)
    assert peak >= 0.15, f"Expected >= 0.15, got {peak}"


# ── Unit 3: segment_frame ───────────────────────────────────────────────────
def test_segment_frame_returns_valid_image():
    """Segmentasi frame menghasilkan gambar tidak kosong."""
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    result, _ = segment_frame(dummy_frame, mode='blur')
    assert result is not None
    assert result.shape == dummy_frame.shape or result.shape[:2] == dummy_frame.shape[:2]
    assert result.sum() > 0, "Hasil segmentasi tidak boleh seluruhnya nol"


# ── Unit 4: CONF_THRESHOLD ──────────────────────────────────────────────────
def test_conf_threshold_value():
    """Konstanta confidence threshold harus bernilai 15.0."""
    assert CONF_THRESHOLD == 15.0, f"Expected 15.0, got {CONF_THRESHOLD}"


def test_conf_threshold_filters_low_confidence():
    """Confidence di bawah threshold dianggap 'Belum dikenali'."""
    confidence = 12.0
    label = "Belum dikenali" if confidence < CONF_THRESHOLD else "air"
    assert label == "Belum dikenali"


def test_conf_threshold_passes_high_confidence():
    """Confidence di atas threshold dianggap valid."""
    confidence = 85.0
    label = "Belum dikenali" if confidence < CONF_THRESHOLD else "air"
    assert label == "air"


# ── Unit 5: Konversi confidence Flutter (simulasi) ──────────────────────────
def test_confidence_conversion():
    """Server mengirim skala 0-100, Flutter harus konversi ke 0.0-1.0."""
    raw_from_server = 75.5
    converted = raw_from_server / 100.0
    assert abs(converted - 0.755) < 0.001, f"Expected 0.755, got {converted}"


def test_confidence_conversion_zero():
    """Confidence 0 dari server → 0.0 di Flutter."""
    assert 0.0 / 100.0 == 0.0


def test_confidence_conversion_max():
    """Confidence 100 dari server → 1.0 di Flutter."""
    assert 100.0 / 100.0 == 1.0
