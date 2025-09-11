#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import time
from typing import Optional

from piper_sdk import *

MM_TO_UM = 1000      # 毫米 -> 微米
DEG_TO_MDEG = 1000   # 度   -> 毫度
M_TO_UM  = 1000 * 1000  # 米   -> 微米（用于演示中 0.05m -> 50mm -> 50000µm）

def move_to_pose(piper: C_PiperInterface_V2,
                 x_mm: float, y_mm: float, z_mm: float,
                 rx_deg: float, ry_deg: float, rz_deg: float,
                 *,
                 speed: int = 100,
                 gripper_deg: Optional[float] = None,
                 settle_s: float = 0.01) -> None:
    """
    将机械臂移动到目标位姿（人类可读单位：mm / °）
    - piper: 已连接且 Enable 成功的 C_PiperInterface_V2 实例
    - x_mm, y_mm, z_mm: 位置（mm）
    - rx_deg, ry_deg, rz_deg: 姿态（°）
    - speed: MotionCtrl_2 的速度（0-100）
    - gripper_deg: 若提供则同时设置夹爪角度（°），内部换算为毫度
    - settle_s: 下发后短暂等待时间（秒）
    """

    # 选择直线/关节等模式与速度（按你的设备语义，这里保持原值）
    piper.MotionCtrl_2(0x01, 0x00, int(speed), 0x00)

    # 单位换算：mm -> µm；° -> 毫度
    X_um  = round(x_mm  * MM_TO_UM)
    Y_um  = round(y_mm  * MM_TO_UM)
    Z_um  = round(z_mm  * MM_TO_UM)
    RX_md = round(rx_deg * DEG_TO_MDEG)
    RY_md = round(ry_deg * DEG_TO_MDEG)
    RZ_md = round(rz_deg * DEG_TO_MDEG)

    # 下发末端位姿
    piper.EndPoseCtrl(X_um, Y_um, Z_um, RX_md, RY_md, RZ_md)

    # 可选：控制夹爪（角度° -> 毫度）
    if gripper_deg is not None:
        J6_md = round(gripper_deg * DEG_TO_MDEG)
        piper.GripperCtrl(abs(J6_md), 1000, 0x01, 0)

    # 简单的缓冲等待（如有更好的运动完成回调，可替换）
    if settle_s > 0:
        time.sleep(settle_s)

# === 夹爪控制封装 ===

def init_gripper(piper: C_PiperInterface_V2) -> None:
    """初始化夹爪（按照提供 demo 的顺序发送两条指令）。"""
    # 注意：这里的两个模式参数 (0x02, 0x01) 及速度 1000 保持与示例一致
    piper.GripperCtrl(0, 1000, 0x02, 0)
    piper.GripperCtrl(0, 1000, 0x01, 0)


def set_gripper_opening_mm(piper: C_PiperInterface_V2, opening_mm: float, speed: int = 1000, mode: int = 0x01) -> None:
    """
    以张开距离（mm）来设置夹爪开度。
    - opening_mm: 需转换为微米下发；与示例中 0.05m 一致，这里统一使用 mm 输入。
    - speed: 夹爪速度（保持与示例一致默认 1000）。
    - mode: 夹爪模式（默认 0x01）。
    """
    opening_um = round(opening_mm * MM_TO_UM)
    piper.GripperCtrl(abs(opening_um), speed, mode, 0)


def read_gripper_state(piper: C_PiperInterface_V2):
    """获取并返回夹爪状态（直接调用底层接口）。"""
    return piper.GetArmGripperMsgs()


def demo_cycle_gripper(piper: C_PiperInterface_V2,
                       open_distance_m: float = 0.05,
                       interval_count: int = 300,
                       sleep_s: float = 0.005) -> None:
    """
    演示夹爪周期性开合（改写自提供的循环示例）。
    - open_distance_m: 最大张开距离（米），默认 0.05m = 50mm。
    - interval_count: 一个阶段的计数阈值，与示例中 300 对应。
    - sleep_s: 循环内 sleep 间隔。
    注意：该函数为无限循环，需在外部通过 Ctrl+C 终止或放在线程中管理。
    """
    range_um = 0
    count = 0
    while True:
        # 打印当前夹爪状态
        print(read_gripper_state(piper))
        count += 1
        if count == 0:
            print("1-----------")
            range_um = 0
        elif count == interval_count:
            print("2-----------")
            # 0.05m -> 微米；保持与原始示例表达式等价：0.05 * 1000 * 1000
            range_um = round(open_distance_m * M_TO_UM)
        elif count == interval_count * 2:
            print("3-----------")
            range_um = 0
            count = 0
        piper.GripperCtrl(abs(range_um), 1000, 0x01, 0)
        time.sleep(sleep_s)

# === 最简用法示例（含夹爪初始化） ===
if __name__ == "__main__":
    p = C_PiperInterface_V2("can0")
    p.ConnectPort()
    while not p.EnablePiper():
        time.sleep(0.01)

    # 初始化夹爪（对应原始示例前两条 GripperCtrl 指令）
    init_gripper(p)

    # 例：移动到 (57, 0, 465) mm, 姿态 (0, 85, 0) °
    move_to_pose(p, 57.0, 0.0, 465.0, 0.0, 85.0, 0.0, speed=10)
    time.sleep(2)
    # 若需要同时控制夹爪角度（角度 -> 毫度）：
    move_to_pose(p, 57.0, 0.0, 360.0, 0.0, 85.0, 0.0, speed=10, gripper_deg=10.0)

    # 若想直接通过张开距离设置（例如张开 30mm）：
    set_gripper_opening_mm(p, 30.0)

    # 若需要持续演示周期性开合（无限循环，谨慎启用）：
    # demo_cycle_gripper(p, open_distance_m=0.05)
