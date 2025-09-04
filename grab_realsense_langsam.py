#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, argparse, time
import numpy as np
import cv2
from PIL import Image
import pyrealsense2 as rs
from lang_sam import LangSAM


def intrinsics_to_dict(intr):
    return {
        "width": intr.width,
        "height": intr.height,
        "fx": intr.fx,
        "fy": intr.fy,
        "ppx": intr.ppx,
        "ppy": intr.ppy,
        "model": str(intr.model),
        "coeffs": list(intr.coeffs),
    }


def main():
    ap = argparse.ArgumentParser(description="Grab RealSense frames, run LangSAM, and save outputs.")
    ap.add_argument("--prompt", type=str, default="fruit.",
                    help="Text prompt for LangSAM (default: 'wheel.')")
    ap.add_argument("--out", type=str, default='./LangRes',
                    help="Output directory. Default: ./capture_YYYYmmdd_HHMMSS")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    out_dir = args.out or f"./capture_{time.strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(out_dir, exist_ok=True)

    # --- RealSense: 开启管线，采集并将深度对齐到彩色坐标系 ---
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    profile = pipeline.start(config)

    try:
        align = rs.align(rs.stream.color)

        # 简单预热几帧，让自动曝光稳定一点
        for _ in range(5):
            pipeline.wait_for_frames()

        frames = pipeline.wait_for_frames()
        frames = align.process(frames)

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            raise RuntimeError("Failed to get frames from RealSense.")

        # numpy 图像
        color_bgr = np.asanyarray(color_frame.get_data())   # HxWx3, uint8
        depth = np.asanyarray(depth_frame.get_data())       # HxW, uint16 (对齐到彩色)

        # --- 保存彩色、深度 ---
        color_path = os.path.join(out_dir, "color.png")
        depth_path = os.path.join(out_dir, "depth.png")  # 16位PNG
        cv2.imwrite(color_path, color_bgr)
        cv2.imwrite(depth_path, depth)

        # --- 保存内参（彩色 & 深度），以及深度尺度 ---
        active = pipeline.get_active_profile()
        dev = active.get_device()
        depth_scale = dev.first_depth_sensor().get_depth_scale()

        color_intr = rs.video_stream_profile(color_frame.profile).get_intrinsics()
        depth_intr = rs.video_stream_profile(depth_frame.profile).get_intrinsics()

        intrinsics_json = {
            "note": "Depth is aligned to color.",
            "color_intrinsics": intrinsics_to_dict(color_intr),
            "depth_intrinsics": intrinsics_to_dict(depth_intr),
            "depth_scale_meters_per_unit": depth_scale,
        }
        with open(os.path.join(out_dir, "intrinsics.json"), "w", encoding="utf-8") as f:
            json.dump(intrinsics_json, f, ensure_ascii=False, indent=2)

        # --- LangSAM 推理（按照你提供的最简用法） ---
        model = LangSAM()
        image_pil = Image.fromarray(cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)).convert("RGB")
        text_prompt = args.prompt
        res = model.predict([image_pil], [text_prompt])
        print(res)  # 可选：打印完整结果，与你给的示例一致

        # 取第 0 张图的结果，重点关注 masks
        if not res or not isinstance(res, list) or len(res) == 0:
            print("No result returned from LangSAM.")
            return

        pred = res[0]
        masks = pred.get("masks", None)
        scores = pred.get("scores", None)  # 与示例一致同时提供
        mask_scores = pred.get("mask_scores", None)

        if masks is None or len(masks) == 0:
            print("No masks found.")
            return

        # 保存每一个 mask 为单独的 PNG（二值化到 0/255）
        masks = np.asarray(masks)
        os.makedirs(os.path.join(out_dir, "masks"), exist_ok=True)
        for i in range(masks.shape[0]):
            m = masks[i]
            # 统一转成 0/1 掩码
            if m.dtype != np.uint8:
                m = (m > 0.5).astype(np.uint8)

            # 用 Pillow 转成 1bit 黑白图像
            img = Image.fromarray(m * 255).convert("1")
            img.save("./LangRes/mask.png")

        # 也顺手存一份 prompt
        with open(os.path.join(out_dir, "prompt.txt"), "w", encoding="utf-8") as f:
            f.write(text_prompt + "\n")

        print(f"Saved to: {os.path.abspath(out_dir)}")

    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()
