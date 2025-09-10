#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
langsam_server.py
在 LangSAM + Orbbec 环境中运行的 socket 服务端：
- 等待客户端发来 JSON 请求：{"prompt": "...", "out_dir": "...(可选)", "width":1280, "height":800, "fps":30, "transfer":"path|base64"}
- 调用 grab_orbbec_langsam.py 生成 color.png / depth.png / mask.png
- 根据 transfer 方式返回：
  * path  ：返回文件路径
  * base64：返回文件内容（适合两环境不共享磁盘时）
"""
import os
import sys
import json
import socket
import argparse
import subprocess
import base64
from typing import Dict, Any

# 允许从同目录导入 socklib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from socklib import send_json, recv_json  # type: ignore

def file_to_b64(path: str) -> str:
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode('ascii')

def handle_request(req: Dict[str, Any]) -> Dict[str, Any]:
    prompt = str(req.get("prompt", "coke."))
    out_dir = str(req.get("out_dir", "./LangRes"))
    width = int(req.get("width", 1280))
    height = int(req.get("height", 800))
    fps = int(req.get("fps", 30))
    transfer = str(req.get("transfer", "path")).lower()  # path | base64

    # 调用现有脚本（保持原注释及逻辑不变）
    cmd = [
        sys.executable, "grab_orbbec_langsam.py",
        "--prompt", prompt,
        "--out", out_dir,
        "--width", str(width),
        "--height", str(height),
        "--fps", str(fps),
    ]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True, text=True)
        log = proc.stdout
    except subprocess.CalledProcessError as e:
        return {"ok": False, "stage": "grab_orbbec_langsam", "error": e.stdout or str(e)}

    # 默认输出路径（与原脚本一致）
    color_path = os.path.join(out_dir, "color.png")
    depth_path = os.path.join(out_dir, "depth.png")
    # 原脚本保存 mask.png 的相对路径固定为 ./LangRes/mask.png（与 out_dir 相同名时对齐）
    # 为保险起见，优先使用 out_dir/mask.png，如果不存在再尝试 ./LangRes/mask.png
    mask_path_prefer = os.path.join(out_dir, "mask.png")
    mask_path_fallback = os.path.join("./LangRes", "mask.png")
    mask_path = mask_path_prefer if os.path.exists(mask_path_prefer) else mask_path_fallback

    if transfer == "base64":
        payload = {
            "ok": True,
            "prompt": prompt,
            "out_dir": os.path.abspath(out_dir),
            "color_b64": file_to_b64(color_path),
            "depth_b64": file_to_b64(depth_path),
            "mask_b64": file_to_b64(mask_path),
            "log": log,
        }
    else:
        payload = {
            "ok": True,
            "prompt": prompt,
            "out_dir": os.path.abspath(out_dir),
            "color_path": os.path.abspath(color_path),
            "depth_path": os.path.abspath(depth_path),
            "mask_path": os.path.abspath(mask_path),
            "log": log,
        }
    return payload

def serve(host: str, port: int):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, port))
        s.listen(8)
        print(f"[LangSAM-Server] Listening on {host}:{port}")
        while True:
            conn, addr = s.accept()
            with conn:
                try:
                    req = recv_json(conn)
                    resp = handle_request(req)
                except Exception as e:
                    resp = {"ok": False, "stage": "server", "error": repr(e)}
                send_json(conn, resp)

def main():
    ap = argparse.ArgumentParser(description="LangSAM socket server")
    ap.add_argument("--host", default="127.0.0.1", type=str)
    ap.add_argument("--port", default=6001, type=int)
    args = ap.parse_args()
    serve(args.host, args.port)

if __name__ == "__main__":
    main()
