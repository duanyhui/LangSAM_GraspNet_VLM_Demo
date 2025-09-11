#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
orbbec_io.py — Orbbec 相机工具类（pyorbbecsdk v2）

功能：
- 打开/关闭相机（优先硬件 D2C 对齐；失败回退软件对齐）
- 采集一帧并（若需要）软件对齐到彩色
- 转 numpy（BGR uint8 彩色、uint16 深度）
- 保存彩色图、保存深度图（16-bit PNG）
- 导出内参/外参/对齐模式/深度尺度 到 intrinsics.json

用法示例（在你的业务代码中直接调用）：
    from orbbec_io import OrbbecCamera

    cam = OrbbecCamera(width=1280, height=800, fps=30)
    cam.start()
    try:
        color_frame, depth_frame = cam.capture(timeout_ms=500)
        color_bgr, depth_raw = cam.frames_to_numpy(color_frame, depth_frame)

        cam.save_color("color.png", color_bgr)
        cam.save_depth16("depth.png", depth_raw)
        cam.export_intrinsics_json("intrinsics.json", depth_frame=depth_frame)
    finally:
        cam.stop()

依赖：
    pip install pyorbbecsdk opencv-python pillow numpy
"""

from __future__ import annotations
import os
import json
import numpy as np
import cv2

# --- Orbbec SDK ---
from pyorbbecsdk import (
    Pipeline, Config, OBSensorType, OBFormat, OBAlignMode,
    OBStreamType, AlignFilter
)

# --- 可选：官方 utils（若在示例目录内可用） ---
try:
    from utils import frame_to_bgr_image  # pyorbbecsdk 官方示例提供
    _HAS_UTILS = True
except Exception:
    _HAS_UTILS = False


class OrbbecCamera:
    """Orbbec 相机工具类：打开/采集/保存/导出参数一条龙。"""

    def __init__(self, width: int = 1280, height: int = 800, fps: int = 30):
        self.width = width
        self.height = height
        self.fps = fps

        self.pipeline: Pipeline | None = None
        self.config: Config | None = None
        self.color_profile = None
        self.depth_profile = None
        self.use_hw_d2c: bool = False
        self.align_filter: AlignFilter | None = None

    # ---------- 生命周期 ----------
    def start(self):
        """启动相机，优先启用硬件 D2C，对齐失败回退软件对齐。"""
        self.pipeline = Pipeline()
        self.config = Config()

        # 选择 Color Profile
        self.color_profile = self._select_color_profile(self.width, self.height, self.fps)

        # 优先硬件 D2C：为 color_profile 查询兼容的 depth_profile
        self.use_hw_d2c = False
        self.depth_profile = None
        try:
            d2c_list = self.pipeline.get_d2c_depth_profile_list(self.color_profile, OBAlignMode.HW_MODE)
            if len(d2c_list) > 0:
                self.depth_profile = d2c_list[0]
                self.config.enable_stream(self.color_profile)
                self.config.enable_stream(self.depth_profile)
                self.config.set_align_mode(OBAlignMode.HW_MODE)
                self.use_hw_d2c = True
        except Exception:
            pass

        # 回退：独立流 + 软件对齐
        if not self.use_hw_d2c:
            self.depth_profile = self.depth_profile or self._select_depth_profile(self.width, self.height, self.fps)
            self.config.enable_stream(self.color_profile)
            self.config.enable_stream(self.depth_profile)
            try:
                from pyorbbecsdk import OBFrameAggregateOutputMode
                self.config.set_frame_aggregate_output_mode(OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
            except Exception:
                pass

        # 启动流
        self.pipeline.start(self.config)

        # 软件对齐器仅在需要时创建
        if not self.use_hw_d2c:
            self.align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
        else:
            self.align_filter = None

        # 预热
        for _ in range(5):
            self.pipeline.wait_for_frames(100)

    def stop(self):
        """停止相机。"""
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except Exception:
                pass
        self.pipeline = None
        self.config = None
        self.color_profile = None
        self.depth_profile = None
        self.align_filter = None
        self.use_hw_d2c = False

    # ---------- 采集与转换 ----------
    def capture(self, timeout_ms: int = 500):
        """采集一组（已对齐到彩色坐标系的）Color/Depth 帧。"""
        if self.pipeline is None:
            raise RuntimeError("Camera not started. Call start() first.")

        frames = self.pipeline.wait_for_frames(timeout_ms)
        if frames is None:
            raise RuntimeError("Failed to get frames from Orbbec camera.")

        if self.align_filter is not None:
            aligned = self.align_filter.process(frames)
            if not aligned:
                raise RuntimeError("AlignFilter returned empty result.")
            frames = aligned.as_frame_set()

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if color_frame is None or depth_frame is None:
            raise RuntimeError("Missing color or depth frame.")

        return color_frame, depth_frame

    def frames_to_numpy(self, color_frame, depth_frame):
        """将 Orbbec 帧转换为 numpy：BGR uint8（HxWx3）、Depth uint16（HxW）。"""
        color_bgr = self._color_frame_to_bgr(color_frame)
        depth_raw = self._depth_frame_to_raw16(depth_frame)
        return color_bgr, depth_raw

    def grab_numpy(self, timeout_ms: int = 500):
        """一步到位：采集并返回 (color_bgr, depth_raw)。"""
        color_f, depth_f = self.capture(timeout_ms)
        return self.frames_to_numpy(color_f, depth_f)

    # ---------- 保存 ----------
    @staticmethod
    def save_color(path: str, color_bgr: np.ndarray):
        """保存 BGR 彩色图（uint8）。"""
        OrbbecCamera._ensure_dir(path)
        if not cv2.imwrite(path, color_bgr):
            raise RuntimeError(f"Failed to write color image: {path}")

    @staticmethod
    def save_depth16(path: str, depth_raw: np.ndarray):
        """保存原始 16-bit 深度为 PNG；不做缩放（单位保持为相机原始单位）。"""
        if depth_raw.dtype != np.uint16:
            raise ValueError("depth_raw must be uint16.")
        OrbbecCamera._ensure_dir(path)
        if not cv2.imwrite(path, depth_raw):
            raise RuntimeError(f"Failed to write depth image: {path}")

    def export_intrinsics_json(self, path: str, depth_frame=None):
        """导出 intrinsics.json（内/外参、对齐模式、深度尺度、Profile 字符串）。"""
        OrbbecCamera._ensure_dir(path)
        intr_json = {
            "note": "Depth aligned to color" if (self.use_hw_d2c or self.align_filter) else "No alignment applied",
            "align_mode": "HW_D2C" if self.use_hw_d2c else ("SW_D2C" if self.align_filter else "NONE"),
            "color_intrinsics": None,
            "depth_intrinsics": None,
            "extrinsic_depth_to_color": None,
            "depth_scale": None,
            "profiles": {
                "color": str(self.color_profile),
                "depth": str(self.depth_profile),
            },
        }
        try:
            c_intr = self.color_profile.get_intrinsic()
            d_intr = self.depth_profile.get_intrinsic()
            intr_json["color_intrinsics"] = self._ob_intrinsic_to_dict(c_intr)
            intr_json["depth_intrinsics"] = self._ob_intrinsic_to_dict(d_intr)
        except Exception:
            pass
        try:
            intr_json["extrinsic_depth_to_color"] = self._ob_extrinsic_to_dict(
                self.depth_profile.get_extrinsic_to(self.color_profile)
            )
        except Exception:
            pass
        if depth_frame is not None:
            try:
                intr_json["depth_scale"] = depth_frame.get_depth_scale()
            except Exception:
                pass

        with open(path, "w", encoding="utf-8") as f:
            json.dump(intr_json, f, ensure_ascii=False, indent=2)

    # ---------- 内部工具：Profile 选择 ----------
    def _select_color_profile(self, width: int, height: int, fps: int):
        plist = Pipeline().get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        try:
            return plist.get_video_stream_profile(width, height, OBFormat.RGB, fps)
        except Exception:
            pass
        try:
            return plist.get_video_stream_profile(0, 0, OBFormat.RGB, 0)
        except Exception:
            pass
        return plist.get_default_video_stream_profile()

    def _select_depth_profile(self, width: int, height: int, fps: int):
        plist = Pipeline().get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        try:
            return plist.get_video_stream_profile(width, height, OBFormat.Y16, fps)
        except Exception:
            pass
        try:
            return plist.get_video_stream_profile(0, 0, OBFormat.Y16, 0)
        except Exception:
            pass
        return plist.get_default_video_stream_profile()

    # ---------- 内部工具：数据转换 ----------
    @staticmethod
    def _color_frame_to_bgr(frame):
        """Color VideoFrame -> BGR (HxWx3, uint8)。优先官方 utils，其次 RGB/MJPG 兜底。"""
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

        # 兜底：尝试直接三通道
        arr = np.frombuffer(data, dtype=np.uint8)
        try:
            return arr.reshape((h, w, 3))
        except Exception as e:
            raise RuntimeError(f"Unsupported color format: {fmt}, error: {e}")

    @staticmethod
    def _depth_frame_to_raw16(frame):
        """Depth VideoFrame -> raw uint16 (HxW)。"""
        if frame is None:
            return None
        vf = frame.as_video_frame()
        w, h = vf.get_width(), vf.get_height()
        data = np.frombuffer(vf.get_data(), dtype=np.uint16)
        return data.reshape((h, w))

    # ---------- 内部工具：序列化 ----------
    @staticmethod
    def _ob_intrinsic_to_dict(intr):
        if intr is None:
            return None
        out = {}
        for k in ("fx", "fy", "cx", "cy", "width", "height"):
            if hasattr(intr, k):
                out[k] = getattr(intr, k)
        if hasattr(intr, "coeffs"):
            try:
                out["coeffs"] = list(intr.coeffs)
            except Exception:
                pass
        if not out:
            out["repr"] = str(intr)
        return out

    @staticmethod
    def _ob_extrinsic_to_dict(ext):
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

    # ---------- 小工具 ----------
    @staticmethod
    def _ensure_dir(path: str):
        """确保路径所在目录存在。"""
        d = os.path.dirname(os.path.abspath(path))
        if d and not os.path.exists(d):
            os.makedirs(d, exist_ok=True)

    # ---------- 可选便捷属性 ----------
    @property
    def align_mode(self) -> str:
        """返回当前对齐模式：'HW_D2C' / 'SW_D2C' / 'NONE'。"""
        return "HW_D2C" if self.use_hw_d2c else ("SW_D2C" if self.align_filter else "NONE")


# ====== 测试主函数：保存一帧并显示 ======
import os
import time
import cv2
import numpy as np

def _depth_to_vis(depth_raw: np.ndarray) -> np.ndarray:
    """
    将 uint16 深度图转为可视化的 8bit 伪彩色图，仅用于显示。
    不改变原始数据的物理意义。
    """
    if depth_raw is None or depth_raw.size == 0:
        return None
    # 简单拉伸到 0~255，再伪彩
    d = depth_raw.copy()
    # 可选：裁掉极端远/近的 0 值
    valid = d > 0
    if np.any(valid):
        vmin = np.percentile(d[valid], 5)
        vmax = np.percentile(d[valid], 95)
        vmax = max(vmax, vmin + 1)
        d = np.clip((d - vmin) * 255.0 / (vmax - vmin), 0, 255).astype(np.uint8)
    else:
        d = cv2.convertScaleAbs(d, alpha=255.0/65535.0)
    vis = cv2.applyColorMap(d, cv2.COLORMAP_JET)
    return vis

def test_capture_once(
    out_dir: str = None,
    width: int = 1280,
    height: int = 800,
    fps: int = 30,
):
    """
    启动相机，抓取 1 帧：
    - 保存：color.png（BGR）、depth.png（16-bit）、intrinsics.json
    - 显示：Color 窗口、Depth 窗口（伪彩）
    """
    from orbbec_io import OrbbecCamera  # 确保 orbbec_io.py 在 Python 路径下

    if out_dir is None:
        out_dir = os.path.abspath(f"./capture_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out_dir, exist_ok=True)

    cam = OrbbecCamera(width=width, height=height, fps=fps)
    cam.start()
    print(f"[INFO] Align mode: {cam.align_mode}")

    try:
        # 抓取一帧
        color_frame, depth_frame = cam.capture(timeout_ms=2800)
        color_bgr, depth_raw = cam.frames_to_numpy(color_frame, depth_frame)

        # 保存
        color_path = os.path.join(out_dir, "color.png")
        depth_path = os.path.join(out_dir, "depth.png")
        intr_path  = os.path.join(out_dir, "intrinsics.json")

        cam.save_color(color_path, color_bgr)
        cam.save_depth16(depth_path, depth_raw)
        cam.export_intrinsics_json(intr_path, depth_frame=depth_frame)

        print(f"[SAVED] color  -> {color_path}")
        print(f"[SAVED] depth  -> {depth_path} (uint16 PNG)")
        print(f"[SAVED] intrin -> {intr_path}")

        # 显示
        depth_vis = _depth_to_vis(depth_raw)

        cv2.imshow("Color (BGR)", color_bgr)
        if depth_vis is not None:
            cv2.imshow("Depth (vis)", depth_vis)

        print("[INFO] 按任意键关闭窗口并退出。")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    finally:
        cam.stop()
        print("[INFO] Camera stopped.")

# 直接运行本文件时执行一次测试；若作为模块导入则不自动运行
if __name__ == "__main__":
    test_capture_once()
