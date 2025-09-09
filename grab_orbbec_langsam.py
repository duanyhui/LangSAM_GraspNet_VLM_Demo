#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Grab frames from an Orbbec camera with pyorbbecsdk, optionally align depth->color,
run LangSAM on the color frame, and save everything (color/depth/intrinsics/masks).

参考 recorder.py 的流式与录制思路；接口遵循 pyorbbecsdk v2 文档。

Usage:
  python3 grab_orbbec_langsam.py \
      --prompt "apple" \
      --out ./LangRes \
      --width 1280 --height 800 --fps 30

Notes:
- 优先尝试硬件 D2C 对齐（OBAlignMode.HW_MODE），否则自动回退到软件对齐（AlignFilter）。
- 如果设备/分辨率不支持指定参数，会自动回退到默认 Profile。
- intrinsics.json 会尽力导出内外参；若 SDK 字段不可用则以字符串形式保存。
"""

import os, json, time, argparse
import numpy as np
import cv2
from PIL import Image

# --- Orbbec SDK ---
from pyorbbecsdk import (
    Pipeline, Config, OBSensorType, OBFormat, OBAlignMode,
    OBFrameType, OBStreamType, AlignFilter
)

# --- LangSAM ---
from lang_sam import LangSAM

# --- utils.frame_to_bgr_image (if available) ---
try:
    from utils import frame_to_bgr_image  # provided in pyorbbecsdk examples
    _HAS_UTILS = True
except Exception:
    _HAS_UTILS = False


def color_frame_to_bgr(frame):
    """Convert Orbbec color VideoFrame to BGR (HxWx3, uint8).
    优先使用官方 utils.frame_to_bgr_image；若不可用，则处理 RGB/MJPG 常见格式。
    """
    if frame is None:
        return None
    if _HAS_UTILS:
        try:
            return frame_to_bgr_image(frame)
        except Exception:
            pass
    vf = frame.as_video_frame()
    w, h = vf.get_width(), vf.get_height()
    fmt = vf.get_format()
    data = vf.get_data()
    if fmt == OBFormat.RGB:
        arr = np.frombuffer(data, dtype=np.uint8).reshape((h, w, 3))
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    if fmt == OBFormat.MJPG:
        arr = np.frombuffer(data, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError("Failed to decode MJPG color frame")
        return bgr
    # 兜底：按三通道 uint8 解读
    arr = np.frombuffer(data, dtype=np.uint8)
    try:
        return arr.reshape((h, w, 3))
    except Exception as e:
        raise RuntimeError(f"Unsupported color format: {fmt}, error: {e}")


def depth_frame_to_raw16(frame):
    """Return raw 16-bit depth image (HxW, uint16), no scaling."""
    if frame is None:
        return None
    vf = frame.as_video_frame()
    w, h = vf.get_width(), vf.get_height()
    data = np.frombuffer(vf.get_data(), dtype=np.uint16)
    depth = data.reshape((h, w))
    return depth


def ob_intrinsic_to_dict(intr):
    """Best-effort serialization of OBIntrinsic-like objects."""
    if intr is None:
        return None
    out = {}
    for k in ("fx", "fy", "cx", "cy", "width", "height"):
        if hasattr(intr, k):
            out[k] = getattr(intr, k)
    # coeffs / distortion params
    if hasattr(intr, "coeffs"):
        try:
            out["coeffs"] = list(intr.coeffs)
        except Exception:
            pass
    if not out:  # fallback to string
        out["repr"] = str(intr)
    return out


def ob_extrinsic_to_dict(ext):
    if ext is None:
        return None
    out = {}
    for k in ("rotation", "translation"):
        if hasattr(ext, k):
            v = getattr(ext, k)
            try:
                out[k] = list(v)
            except Exception:
                out[k] = str(v)
    if not out:
        out["repr"] = str(ext)
    return out


def select_color_profile(pipeline, width, height, fps):
    plist = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    # try exact RGB match
    try:
        return plist.get_video_stream_profile(width, height, OBFormat.RGB, fps)
    except Exception:
        pass
    # any RGB
    try:
        return plist.get_video_stream_profile(0, 0, OBFormat.RGB, 0)
    except Exception:
        pass
    # default
    return plist.get_default_video_stream_profile()


def select_depth_profile(pipeline, width, height, fps):
    plist = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
    # try exact Y16 match
    try:
        return plist.get_video_stream_profile(width, height, OBFormat.Y16, fps)
    except Exception:
        pass
    # any Y16
    try:
        return plist.get_video_stream_profile(0, 0, OBFormat.Y16, 0)
    except Exception:
        pass
    # default
    return plist.get_default_video_stream_profile()


def main():
    ap = argparse.ArgumentParser(description="Grab Orbbec frames, run LangSAM, and save outputs.")
    ap.add_argument("--prompt", type=str, default="mouse.", help="Text prompt for LangSAM")
    ap.add_argument("--out", type=str, default="./LangRes", help="Output directory")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=800)
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    out_dir = args.out or f"./capture_{time.strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(out_dir, exist_ok=True)

    # --- set up pipeline & config ---
    pipeline = Pipeline()
    config = Config()

    # choose color profile first
    color_profile = select_color_profile(pipeline, args.width, args.height, args.fps)

    # try Hardware D2C (depth aligned to color)
    use_hw_d2c = False
    depth_profile = None
    try:
        d2c_list = pipeline.get_d2c_depth_profile_list(color_profile, OBAlignMode.HW_MODE)
        if len(d2c_list) > 0:
            depth_profile = d2c_list[0]
            config.enable_stream(color_profile)
            config.enable_stream(depth_profile)
            config.set_align_mode(OBAlignMode.HW_MODE)
            use_hw_d2c = True
    except Exception:
        pass

    # fallback to independent profiles (software align later)
    align_filter = None
    if not use_hw_d2c:
        depth_profile = depth_profile or select_depth_profile(pipeline, args.width, args.height, args.fps)
        config.enable_stream(color_profile)
        config.enable_stream(depth_profile)
        # FULL_FRAME_REQUIRE: 确保帧集中同时包含 color & depth
        try:
            from pyorbbecsdk import OBFrameAggregateOutputMode
            config.set_frame_aggregate_output_mode(OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
        except Exception:
            pass

    # start stream
    pipeline.start(config)

    # prepare software aligner only if needed
    if not use_hw_d2c:
        align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)

    # warmup
    for _ in range(5):
        pipeline.wait_for_frames(100)

    # get one frameset (aligned if needed)
    frames = pipeline.wait_for_frames(200)
    if frames is None:
        raise RuntimeError("Failed to get frames from Orbbec camera.")

    if align_filter is not None:
        aligned = align_filter.process(frames)
        if not aligned:
            raise RuntimeError("AlignFilter returned empty result.")
        frames = aligned.as_frame_set()

    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()
    if color_frame is None or depth_frame is None:
        raise RuntimeError("Missing color or depth frame.")

    # --- convert to numpy ---
    color_bgr = color_frame_to_bgr(color_frame)   # HxWx3, uint8
    depth_raw = depth_frame_to_raw16(depth_frame) # HxW, uint16 (aligned to color if D2C enabled)

    # --- save images ---
    color_path = os.path.join(out_dir, "color.png")
    depth_path = os.path.join(out_dir, "depth.png")  # 16-bit PNG
    cv2.imwrite(color_path, color_bgr)
    cv2.imwrite(depth_path, depth_raw)

    # --- save intrinsics/extrinsics & depth scale ---
    intr_json = {
        "note": "Depth is aligned to color" if use_hw_d2c or align_filter else "No alignment applied",
        "align_mode": "HW_D2C" if use_hw_d2c else ("SW_D2C" if align_filter else "NONE"),
        "color_intrinsics": None,
        "depth_intrinsics": None,
        "extrinsic_depth_to_color": None,
        "depth_scale": None,
        "profiles": {
            "color": str(color_profile),
            "depth": str(depth_profile),
        },
    }
    try:
        c_intr = color_profile.get_intrinsic()
        d_intr = depth_profile.get_intrinsic()
        intr_json["color_intrinsics"] = ob_intrinsic_to_dict(c_intr)
        intr_json["depth_intrinsics"] = ob_intrinsic_to_dict(d_intr)
    except Exception:
        pass
    try:
        intr_json["extrinsic_depth_to_color"] = ob_extrinsic_to_dict(depth_profile.get_extrinsic_to(color_profile))
    except Exception:
        pass
    try:
        intr_json["depth_scale"] = depth_frame.get_depth_scale()
    except Exception:
        pass
    with open(os.path.join(out_dir, "intrinsics.json"), "w", encoding="utf-8") as f:
        json.dump(intr_json, f, ensure_ascii=False, indent=2)

    # --- LangSAM inference ---
    model = LangSAM()
    image_pil = Image.fromarray(cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)).convert("RGB")
    res = model.predict([image_pil], [args.prompt])

    masks_dir = os.path.join(out_dir, "masks")
    os.makedirs(masks_dir, exist_ok=True)

    if isinstance(res, list) and len(res) > 0 and isinstance(res[0], dict):
        pred = res[0]
        masks = pred.get("masks", None)
        if masks is not None and len(masks) > 0:
            masks = np.asarray(masks)
            # save each mask as 0/255 PNG
            for i in range(masks.shape[0]):
                m = masks[i]
                # 统一转成 0/1 掩码
                if m.dtype != np.uint8:
                    m = (m > 0.5).astype(np.uint8)

                # 用 Pillow 转成 1bit 黑白图像
                img = Image.fromarray(m * 255).convert("1")
                img.save("./LangRes/mask.png")

            # optional: quick overlay for visualization
            overlay = color_bgr.copy()
            mask_any = (masks.max(axis=0) > 0.5).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_any, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
            cv2.imwrite(os.path.join(out_dir, "overlay.png"), overlay)

    # --- save prompt ---
    with open(os.path.join(out_dir, "prompt.txt"), "w", encoding="utf-8") as f:
        f.write(args.prompt + "\n")

    print(f"Saved results to: {os.path.abspath(out_dir)}")

    # cleanup
    pipeline.stop()


if __name__ == "__main__":
    main()
