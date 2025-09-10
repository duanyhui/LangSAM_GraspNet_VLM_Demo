#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
graspnet_server.py
在 GraspNet 环境中运行的 socket 服务端：
- 等待客户端发来 JSON 请求：
  {"color_path": "...", "depth_path": "...", "mask_path": "..."} 或使用 base64 传输
- 动态导入本地的 GraspNetDemo.py 并调用 run_grasp_inference(...) 完成推理
- 返回最优抓取位姿与宽度等数值结果，Open3D 可视化窗口会在此环境中弹出
"""
import os
import sys
import json
import socket
import argparse
import base64
import tempfile
from typing import Dict, Any, Tuple

# 允许从同目录导入 socklib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from socklib import send_json, recv_json  # type: ignore

def b64_to_file(b64: str, path: str):
    import base64
    data = base64.b64decode(b64.encode('ascii'))
    with open(path, 'wb') as f:
        f.write(data)

def import_graspnetdemo(module_path: str):
    """
    动态导入任意路径下的 GraspNetDemo.py，返回模块对象
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location("GraspNetDemo", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import GraspNetDemo from: {module_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def handle_request(req: Dict[str, Any], graspnetdemo_path: str) -> Dict[str, Any]:
    # 解析输入：既支持 path 也支持 base64
    use_b64 = bool(req.get("use_b64", False))

    if use_b64:
        tmpdir = tempfile.mkdtemp(prefix="graspnet_inputs_")
        color_path = os.path.join(tmpdir, "color.png")
        depth_path = os.path.join(tmpdir, "depth.png")
        mask_path  = os.path.join(tmpdir, "mask.png")
        b64_to_file(req["color_b64"], color_path)
        b64_to_file(req["depth_b64"], depth_path)
        b64_to_file(req.get("mask_b64", ""), mask_path)
    else:
        color_path = str(req["color_path"])
        depth_path = str(req["depth_path"])
        mask_path  = str(req.get("mask_path", ""))

    # 导入并调用现有推理函数（保持原注释及逻辑不变）
    Demo = import_graspnetdemo(graspnetdemo_path)
    if not hasattr(Demo, "run_grasp_inference"):
        raise AttributeError("GraspNetDemo.py 中未找到 run_grasp_inference 函数")

    t, R, w = Demo.run_grasp_inference(color_path, depth_path, sam_mask_path=mask_path if mask_path else None)

    return {
        "ok": True,
        "best_translation": [float(x) for x in t],
        "best_rotation": [[float(v) for v in row] for row in R],
        "best_width": float(w),
        "note": "Open3D 可视化窗口在本环境中显示；如需关闭请在窗口中按 q."
    }

def serve(host: str, port: int, graspnetdemo_path: str):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, port))
        s.listen(8)
        print(f"[GraspNet-Server] Listening on {host}:{port}")
        print(f"[GraspNet-Server] Using GraspNetDemo at: {graspnetdemo_path}")
        while True:
            conn, addr = s.accept()
            with conn:
                try:
                    req = recv_json(conn)
                    resp = handle_request(req, graspnetdemo_path)
                except Exception as e:
                    resp = {"ok": False, "stage": "graspnet", "error": repr(e)}
                send_json(conn, resp)

def main():
    ap = argparse.ArgumentParser(description="GraspNet socket server")
    ap.add_argument("--host", default="127.0.0.1", type=str)
    ap.add_argument("--port", default=6002, type=int)
    ap.add_argument("--graspnetdemo", default="GraspNetDemo.py", type=str,
                    help="GraspNetDemo.py 的绝对或相对路径")
    args = ap.parse_args()
    serve(args.host, args.port, args.graspnetdemo)

if __name__ == "__main__":
    main()
