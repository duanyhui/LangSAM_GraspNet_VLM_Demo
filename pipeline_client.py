#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pipeline_client.py
一个简单的编排客户端：
- 连接 LangSAM 服务器，发送 prompt，等待生成结果
- 将生成的 color/depth/mask 输入发送给 GraspNet 服务器，等待抓取结果
- 打印最终的数值化抓取姿态信息
"""
import os
import sys
import argparse
import socket
from typing import Dict, Any

# 允许从同目录导入 socklib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from socklib import send_json, recv_json  # type: ignore

def request(host: str, port: int, payload: Dict[str, Any]) -> Dict[str, Any]:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        send_json(s, payload)
        return recv_json(s)

def main():
    ap = argparse.ArgumentParser(description="LangSAM->GraspNet 编排客户端")
    ap.add_argument("--prompt", type=str, required=True, help="物体文本提示词")
    ap.add_argument("--ls-host", default="127.0.0.1", type=str)
    ap.add_argument("--ls-port", default=6001, type=int)
    ap.add_argument("--gn-host", default="127.0.0.1", type=str)
    ap.add_argument("--gn-port", default=6002, type=int)
    ap.add_argument("--transfer", default="path", choices=["path", "base64"],
                    help="不同环境是否共享磁盘：共享用 path，不共享用 base64")
    ap.add_argument("--out-dir", default="./LangRes", type=str)
    ap.add_argument("--width", default=1280, type=int)
    ap.add_argument("--height", default=800, type=int)
    ap.add_argument("--fps", default=30, type=int)
    args = ap.parse_args()

    print(f"[*] 请求 LangSAM 生成: prompt='{args.prompt}' ...")
    req_ls = {
        "prompt": args.prompt,
        "out_dir": args.out_dir,
        "width": args.width,
        "height": args.height,
        "fps": args.fps,
        "transfer": args.transfer
    }
    resp_ls = request(args.ls_host, args.ls_port, req_ls)
    if not resp_ls.get("ok"):
        print("[LangSAM ERROR]", resp_ls)
        sys.exit(2)

    print("[*] LangSAM 完成，开始调用 GraspNet ...")
    if args.transfer == "base64":
        req_gn = {
            "use_b64": True,
            "color_b64": resp_ls["color_b64"],
            "depth_b64": resp_ls["depth_b64"],
            "mask_b64": resp_ls.get("mask_b64", ""),
        }
    else:
        req_gn = {
            "use_b64": False,
            "color_path": resp_ls["color_path"],
            "depth_path": resp_ls["depth_path"],
            "mask_path": resp_ls.get("mask_path", ""),
        }

    resp_gn = request(args.gn_host, args.gn_port, req_gn)
    if not resp_gn.get("ok"):
        print("[GraspNet ERROR]", resp_gn)
        sys.exit(3)

    print("\n=== 最优抓取结果（来自 GraspNet 环境） ===")
    print("Translation (x, y, z):", resp_gn["best_translation"])
    print("Rotation (3x3):")
    for row in resp_gn["best_rotation"]:
        print(" ", row)
    print("Width:", resp_gn["best_width"])
    print("\n提示：Open3D 可视化窗口已在 GraspNet 环境中弹出。")

if __name__ == "__main__":
    main()
