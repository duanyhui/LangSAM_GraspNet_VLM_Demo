#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
socklib.py
一个简洁稳健的 TCP JSON 消息协议工具：采用 4 字节网络序长度 + JSON UTF-8 字节流，避免粘包/拆包。
"""
import json
import socket
import struct
from typing import Any, Dict, Tuple

def send_json(sock: socket.socket, obj: Dict[str, Any]) -> None:
    """
    将 Python 对象编码为 JSON，并以 4 字节长度前缀 + 数据的形式发送。
    """
    data = json.dumps(obj, ensure_ascii=False).encode('utf-8')
    header = struct.pack('!I', len(data))
    sock.sendall(header + data)

def recv_exactly(sock: socket.socket, n: int) -> bytes:
    """
    从套接字中恰好读取 n 字节，否则抛出异常。
    """
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Socket closed while receiving data")
        buf.extend(chunk)
    return bytes(buf)

def recv_json(sock: socket.socket) -> Dict[str, Any]:
    """
    接收一条 4 字节长度前缀的 JSON 消息，并解码为 Python 对象。
    """
    header = recv_exactly(sock, 4)
    (length,) = struct.unpack('!I', header)
    payload = recv_exactly(sock, length)
    return json.loads(payload.decode('utf-8'))
