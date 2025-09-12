#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
handeye_pipeline.py  (修复 & 函数化版本)

功能（函数调用）：
  - capture_handeye(...)  采集多组样本（相机检测棋盘 + 读取机械臂 EE 位姿），保存数据集 JSON
  - solve_handeye(...)    使用 OpenCV Robot-World/Hand-Eye 求解，支持 eye_to_hand / eye_in_hand
  - verify_handeye(...)   按所选安装模式做几何一致性验证（位置/角度误差）

与原脚本的不同：
  - 不再用 argparse/子命令；改为直接函数调用
  - 采集逻辑（窗口/按键/阈值/相机接口）保持不变
  - 修复：calibrateRobotWorldHandEye 的入参顺序与方向、输出语义匹配、verify 等式与模式一致
"""

from __future__ import annotations
import os, json, time
import numpy as np
import cv2

# === 你的现有接口 ===
from orbbec_io import OrbbecCamera      # 相机采集/导出内参
from piper_sdk import C_PiperInterface_V2

# ---------------- 基础数学/工具 ----------------
def rvec_tvec_to_T(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4, dtype=np.float64)
    T[:3,:3] = R
    T[:3, 3] = tvec.reshape(3)
    return T

def T_to_rt(T):
    R = T[:3,:3]
    t = T[:3, 3].reshape(3,1)
    rvec, _ = cv2.Rodrigues(R)
    return rvec, t

def make_chessboard_object_points(cols, rows, square_mm):
    """内角点坐标（单位 m；z=0）"""
    objp = np.zeros((rows*cols, 3), np.float32)
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)  # 列优先
    objp[:,:2] = grid * (square_mm / 1000.0)
    return objp

def inv_T(T):
    """SE(3) 逆"""
    R = T[:3,:3]; t = T[:3,3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3,:3] = R.T
    Ti[:3,3]  = -R.T @ t
    return Ti

def average_rotations(R_list):
    """用SVD做最近正交投影的旋转平均"""
    M = np.zeros((3,3), dtype=np.float64)
    for R in R_list:
        M += R
    U, _, Vt = np.linalg.svd(M)
    R_avg = U @ Vt
    if np.linalg.det(R_avg) < 0:
        U[:, -1] *= -1
        R_avg = U @ Vt
    return R_avg

def se3_from_Rt(R, t):
    T = np.eye(4, dtype=np.float64)
    T[:3,:3] = R
    T[:3, 3] = t.reshape(3)
    return T

# ---------------- Piper 末端位姿读取 ----------------
def get_end_pose_data(piper, pos_unit_raw="mm", ang_unit_raw="deg"):
    """
    仅获取机械臂末端位姿数据，不打印
    返回六元组（mm/deg）
    """
    end_pose_msg = piper.GetArmEndPoseMsgs()
    x_raw = end_pose_msg.end_pose.X_axis
    y_raw = end_pose_msg.end_pose.Y_axis
    z_raw = end_pose_msg.end_pose.Z_axis
    rx_raw = end_pose_msg.end_pose.RX_axis
    ry_raw = end_pose_msg.end_pose.RY_axis
    rz_raw = end_pose_msg.end_pose.RZ_axis

    x_mm = (x_raw / 1000.0) if pos_unit_raw == "um"  else float(x_raw)
    y_mm = (y_raw / 1000.0) if pos_unit_raw == "um"  else float(y_raw)
    z_mm = (z_raw / 1000.0) if pos_unit_raw == "um"  else float(z_raw)
    rx_deg = (rx_raw / 1000.0) if ang_unit_raw == "mdeg" else float(rx_raw)
    ry_deg = (ry_raw / 1000.0) if ang_unit_raw == "mdeg" else float(ry_raw)
    rz_deg = (rz_raw / 1000.0) if ang_unit_raw == "mdeg" else float(rz_raw)

    return x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg

def get_ee_pose_from_sdk(piper: C_PiperInterface_V2,
                         pos_unit_raw="mm", ang_unit_raw="deg", euler_order="XYZ"):
    """
    返回 T_base_ee（单位 m），旋转按控制器欧拉角顺序（XYZ/ZYX）
    """
    X,Y,Z,RX,RY,RZ = get_end_pose_data(piper, pos_unit_raw, ang_unit_raw)
    print("[DEBUG] 机械臂末端位姿（mm/deg）:",
          f"X={X:.1f}, Y={Y:.1f}, Z={Z:.1f}, RX={RX:.1f}, RY={RY:.1f}, RZ={RZ:.1f}")

    p = np.array([X, Y, Z], dtype=np.float64) / 1000.0
    r = np.radians([RX, RY, RZ])

    Rx = np.array([[1, 0, 0],
                   [0, np.cos(r[0]), -np.sin(r[0])],
                   [0, np.sin(r[0]),  np.cos(r[0])]])
    Ry = np.array([[np.cos(r[1]), 0, np.sin(r[1])],
                   [0, 1, 0],
                   [-np.sin(r[1]), 0, np.cos(r[1])]])
    Rz = np.array([[np.cos(r[2]), -np.sin(r[2]), 0],
                   [np.sin(r[2]),  np.cos(r[2]), 0],
                   [0, 0, 1]])
    if str(euler_order).upper() == "XYZ":
        R = Rx @ Ry @ Rz
    else:  # ZYX
        R = Rz @ Ry @ Rx

    T = np.eye(4); T[:3,:3] = R; T[:3,3] = p
    return T

# ---------------- 棋盘姿态估计 ----------------
def detect_chessboard_pose(color_bgr, cols, rows, square_mm, K, dist, reproj_thresh=0.4):
    """
    返回 (ok, T_cam_board)，单位米。
    """
    gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
    pattern_size = (cols, rows)

    flag_combinations = [
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FILTER_QUADS,
        cv2.CALIB_CB_ADAPTIVE_THRESH,
        cv2.CALIB_CB_NORMALIZE_IMAGE,
        0
    ]

    ok, corners = False, None
    for flags in flag_combinations:
        ok, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
        if ok:
            # print(f"[DEBUG] 棋盘格检测成功，flags={flags}, 角点={len(corners)}")
            break

    if not ok:
        # print(f"[DEBUG] 棋盘检测失败，期望 {cols}x{rows}={cols*rows} 个内角点")
        return False, None

    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), term)
    objp = make_chessboard_object_points(cols, rows, square_mm)

    ok, rvec, tvec = cv2.solvePnP(objp, corners, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        print("[DEBUG] PnP 求解失败")
        return False, None

    proj, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
    reproj = np.mean(np.linalg.norm(proj.squeeze() - corners.squeeze(), axis=1))
    if reproj > reproj_thresh:
        print(f"[DROP] reproj={reproj:.3f}px")
        return False, None

    T_cam_board = rvec_tvec_to_T(rvec, tvec)
    return True, T_cam_board

# ---------------- 采集（逻辑基本不变） ----------------
def capture_handeye(out="./handeye_run",
                    cols=7, rows=5, square_mm=25.0,
                    width=1280, height=800, fps=30,
                    pos_unit_raw="mm", ang_unit_raw="deg", euler_order="XYZ",
                    reproj_thresh=0.4):
    """
    交互式采集：按 SPACE 采样，q 退出。保存 {out}/handeye_dataset.json
    """
    os.makedirs(out, exist_ok=True)

    cam = OrbbecCamera(width=width, height=height, fps=fps)
    cam.start()
    print(f"[INFO] Orbbec align mode: {cam.align_mode}")

    intr_path = os.path.join(out, "intrinsics.json")
    try:
        cf, df = cam.capture(timeout_ms=6000)
        cam.export_intrinsics_json(intr_path, depth_frame=df)
    except RuntimeError as e:
        print(f"[WARN] 初始采集失败: {e} -> 重试")
        for retry in range(3):
            try:
                cf, df = cam.capture(timeout_ms=3000)
                cam.export_intrinsics_json(intr_path, depth_frame=df)
                break
            except RuntimeError:
                if retry == 2:
                    print("[WARN] 使用仅彩色内参导出")
                    cf, _ = cam.capture(timeout_ms=3000)
                    cam.export_intrinsics_json(intr_path, depth_frame=None)
                time.sleep(0.5)

    c_intr = cam.color_profile.get_intrinsic()
    fx, fy, cx, cy = c_intr.fx, c_intr.fy, c_intr.cx, c_intr.cy
    print("fx, fy, cx, cy =", fx, fy, cx, cy)

    try:
        dist_coeffs = np.array(list(c_intr.coeffs), dtype=np.float64).reshape(-1,1)
    except Exception:
        dist_coeffs = np.zeros((5,1), dtype=np.float64)
    K = np.array([[fx, 0, cx],[0, fy, cy],[0, 0, 1]], dtype=np.float64)

    piper = C_PiperInterface_V2("can0")
    piper.ConnectPort()
    while not piper.EnablePiper():
        time.sleep(0.01)
    print("[INFO] Piper connected & enabled.")

    dataset = {
        "cols": cols, "rows": rows, "square_mm": square_mm,
        "K": K.tolist(), "dist": dist_coeffs.reshape(-1).tolist(),
        "samples": []
    }

    print("\n[CAPTURE] 操作提示：")
    print("  - 移动机械臂让棋盘出现在不同位置/距离/姿态（覆盖视野四角、近中远、多角度）")
    print("  - 每到一个姿态，按 'SPACE' 采集；按 'q' 结束。建议 ≥ 20 组。")

    count = 0
    try:
        while True:
            try:
                time.sleep(0.1)
                cf, df = cam.capture(timeout_ms=3000)
                color_bgr, _ = cam.frames_to_numpy(cf, df)
            except RuntimeError as e:
                print(f"[WARN] 采集失败: {e}, 跳过此帧")
                continue

            vis = color_bgr.copy()
            ok, T_cam_board = detect_chessboard_pose(
                color_bgr, cols, rows, square_mm, K, dist_coeffs, reproj_thresh
            )
            if ok:
                pattern_size = (cols, rows)
                gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
                ret, corners = cv2.findChessboardCorners(gray, pattern_size)
                if ret:
                    cv2.drawChessboardCorners(vis, pattern_size, corners, ret)
                cv2.putText(vis, "DETECTED", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,200,0), 2)
            else:
                cv2.putText(vis, "NOT FOUND", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)

            cv2.imshow("Capture (press SPACE to save, q to quit)", vis)
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            if key == 32 and ok:  # SPACE
                T_base_ee = get_ee_pose_from_sdk(piper, pos_unit_raw, ang_unit_raw, euler_order)
                sample = {
                    "T_cam_board": T_cam_board.tolist(),
                    "T_base_ee": T_base_ee.tolist(),
                    "snapshot": f"cap_{count:03d}.png"
                }
                dataset["samples"].append(sample)
                cv2.imwrite(os.path.join(out, sample["snapshot"]), color_bgr)
                count += 1
                print(f"[SAVED] sample #{count}")
    finally:
        cv2.destroyAllWindows()
        cam.stop()

    data_path = os.path.join(out, "handeye_dataset.json")
    with open(data_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    print(f"\n[DONE] 共采集 {len(dataset['samples'])} 组样本 → {data_path}")
    return data_path

# ---------------- 求解 ----------------
def _build_world2cam_and_base2gripper(D, invert_cam_board_for_world2cam: bool, always_inv_base_ee=True):
    """
    按指定策略构造 RobotWorld/HandEye 的两组观测序列：
    world2cam[], base2gripper[]
    """
    R_world2cam, t_world2cam = [], []
    R_base2gripper, t_base2gripper = [], []

    for s in D["samples"]:
        T_cam_board = np.array(s["T_cam_board"], dtype=np.float64)
        T_base_ee   = np.array(s["T_base_ee"],   dtype=np.float64)

        # world2cam: 由 T_cam_board 或其逆得到
        T_w2c = T_cam_board if not invert_cam_board_for_world2cam else inv_T(T_cam_board)
        R_world2cam.append(T_w2c[:3,:3])
        t_world2cam.append(T_w2c[:3,3].reshape(3,1))

        # base2gripper: 一律 inv(T_base_ee)
        T_b2g = inv_T(T_base_ee) if always_inv_base_ee else T_base_ee
        R_base2gripper.append(T_b2g[:3,:3])
        t_base2gripper.append(T_b2g[:3,3].reshape(3,1))

    return R_world2cam, t_world2cam, R_base2gripper, t_base2gripper

def solve_handeye(data_path: str,
                  out_path: str = "./handeye_calib.json",
                  setup: str = "auto",
                  method: int = cv2.CALIB_ROBOT_WORLD_HAND_EYE_SHAH):
    """
    求解与保存标定：
      setup in {"auto","eye_to_hand","eye_in_hand"}
        - eye_to_hand：外置相机、棋盘固定在夹爪（我们需要 T_base_cam、T_ee_board）
        - eye_in_hand：相机在末端、棋盘固定在世界（输出 T_base_world、T_gripper_cam）
        - auto：基于 T_cam_board 的平移方差做粗略判别（> 1 cm 视为相机相对棋盘有明显运动 → eye_in_hand）
    """
    with open(data_path, "r", encoding="utf-8") as f:
        D = json.load(f)

    # --- 自动模式的简单判别：看 T_cam_board 的平移是否明显变化 ---
    if setup == "auto":
        t_all = [np.array(s["T_cam_board"], dtype=np.float64)[:3,3] for s in D["samples"]]
        if len(t_all) < 3:
            setup = "eye_to_hand"  # 样本太少，默认外置相机
        else:
            t_stack = np.stack(t_all, axis=0)
            std_meter = np.std(t_stack, axis=0)
            setup = "eye_in_hand" if np.linalg.norm(std_meter) > 0.01 else "eye_to_hand"  # 1 cm 阈值
        print(f"[AUTO] 判定安装形态：{setup}")

    # --- 构造观测并调用 OpenCV 求解 ---
    if setup == "eye_to_hand":
        # 关键：把“world”定义为“camera”、把“cam”定义为“board”
        # 因此 world2cam[] 应该是 “cam→board”，即 inv(T_cam_board)
        invert_cam_board_for_world2cam = True   # 用 inv(T_cam_board)
        R_w2c, t_w2c, R_b2g, t_b2g = _build_world2cam_and_base2gripper(
            D, invert_cam_board_for_world2cam=True, always_inv_base_ee=True
        )
        if not hasattr(cv2, "calibrateRobotWorldHandEye"):
            raise RuntimeError("OpenCV 缺少 calibrateRobotWorldHandEye（需要 4.8+）")

        R_base2world, t_base2world, R_gripper2cam, t_gripper2cam = cv2.calibrateRobotWorldHandEye(
            R_w2c, t_w2c, R_b2g, t_b2g, method=method
        )
        # 这里的 world == camera, cam == board
        # 因此：
        #   base2world == base2camera
        #   gripper2cam == ee2board
        T_base_cam = se3_from_Rt(R_base2world, t_base2world)
        T_ee_board = se3_from_Rt(R_gripper2cam, t_gripper2cam)

        out = {
            "mode": "eye_to_hand",
            "T_base_cam": T_base_cam.tolist(),
            "T_ee_board": T_ee_board.tolist(),
            "num_samples": len(D["samples"]),
            "method": int(method),
            "note": "由 calibrateRobotWorldHandEye 通过（world=cam, cam=board）置换得到"
        }

    elif setup == "eye_in_hand":
        # 直接按文档语义：world == board, cam == camera
        invert_cam_board_for_world2cam = False  # 直接用 T_cam_board
        R_w2c, t_w2c, R_b2g, t_b2g = _build_world2cam_and_base2gripper(
            D, invert_cam_board_for_world2cam=False, always_inv_base_ee=True
        )
        if not hasattr(cv2, "calibrateRobotWorldHandEye"):
            raise RuntimeError("OpenCV 缺少 calibrateRobotWorldHandEye（需要 4.8+）")

        R_base2world, t_base2world, R_gripper2cam, t_gripper2cam = cv2.calibrateRobotWorldHandEye(
            R_w2c, t_w2c, R_b2g, t_b2g, method=method
        )
        # world == board, cam == camera
        T_base_world  = se3_from_Rt(R_base2world, t_base2world)   # base->board
        T_gripper_cam = se3_from_Rt(R_gripper2cam, t_gripper2cam) # ee->cam

        out = {
            "mode": "eye_in_hand",
            "T_base_world":  T_base_world.tolist(),
            "T_gripper_cam": T_gripper_cam.tolist(),
            "num_samples": len(D["samples"]),
            "method": int(method)
        }
    else:
        raise ValueError("setup 必须是 {'auto','eye_to_hand','eye_in_hand'}")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[OK] 标定完成：写入 {out_path}  (mode={out['mode']})")
    return out_path

# ---------------- 验证 ----------------
def _rot_angle_deg(R):
    angle = np.degrees(np.arccos(np.clip((np.trace(R)-1)/2.0, -1.0, 1.0)))
    return float(abs(angle))

def verify_handeye(data_path: str, calib_path: str):
    """
    根据 calib.json 中的 'mode' 字段自动选择校验公式。
    打印平均/最大 位置(单位 mm)/角度(单位 deg)误差。
    """
    with open(data_path, "r", encoding="utf-8") as f:
        D = json.load(f)
    with open(calib_path, "r", encoding="utf-8") as f:
        C = json.load(f)

    mode = C.get("mode", "eye_to_hand")

    pos_errs, ang_errs = [], []

    if mode == "eye_to_hand":
        # 期望恒定：T_base_cam, T_ee_board
        T_base_cam = np.array(C["T_base_cam"], dtype=np.float64)
        T_ee_board = np.array(C["T_ee_board"], dtype=np.float64)

        for s in D["samples"]:
            T_cam_board = np.array(s["T_cam_board"], dtype=np.float64)
            T_base_ee   = np.array(s["T_base_ee"],   dtype=np.float64)

            # 预测： base<-board
            T_base_board_pred = T_base_cam @ T_cam_board
            # “真值”： base<-ee<-board
            T_base_board_gt   = T_base_ee @ T_ee_board

            dp = T_base_board_pred[:3,3] - T_base_board_gt[:3,3]
            pos_errs.append(np.linalg.norm(dp) * 1000.0)

            dR = T_base_board_pred[:3,:3].T @ T_base_board_gt[:3,:3]
            ang_errs.append(_rot_angle_deg(dR))

    elif mode == "eye_in_hand":
        # 期望恒定：T_base_world, T_gripper_cam
        T_base_world  = np.array(C["T_base_world"],  dtype=np.float64)  # base<-board
        T_gripper_cam = np.array(C["T_gripper_cam"], dtype=np.float64)  # ee<-cam

        for s in D["samples"]:
            T_cam_board = np.array(s["T_cam_board"], dtype=np.float64)  # cam<-board
            T_base_ee   = np.array(s["T_base_ee"],   dtype=np.float64)  # base<-ee

            # 预测： base<-world ≈ (base<-ee) * (ee<-cam) * (cam<-board)
            T_base_world_pred = (T_base_ee @ T_gripper_cam) @ T_cam_board

            dp = T_base_world_pred[:3,3] - T_base_world[:3,3]
            pos_errs.append(np.linalg.norm(dp) * 1000.0)

            dR = T_base_world_pred[:3,:3].T @ T_base_world[:3,:3]
            ang_errs.append(_rot_angle_deg(dR))
    else:
        raise ValueError("calib['mode'] 必须是 {'eye_to_hand','eye_in_hand'}")

    print(f"[VERIFY] 平均位置误差: {np.mean(pos_errs):.2f} mm  | 最大: {np.max(pos_errs):.2f} mm")
    print(f"[VERIFY] 平均角度误差: {np.mean(ang_errs):.3f} deg | 最大: {np.max(ang_errs):.3f} deg")
    return {
        "mode": mode,
        "mean_pos_mm": float(np.mean(pos_errs)),
        "max_pos_mm":  float(np.max(pos_errs)),
        "mean_ang_deg":float(np.mean(ang_errs)),
        "max_ang_deg": float(np.max(ang_errs)),
        "num_samples": len(D["samples"])
    }

if __name__ == "__main__":
    # 示例调用
    # 1) 采集（与原参数一致）
    data_path = capture_handeye(
        out="run5",
        cols=7, rows=5, square_mm=25,
        pos_unit_raw="um", ang_unit_raw="mdeg",
        euler_order="ZXY",
        reproj_thresh=0.4
    )
    calib_path = solve_handeye(
        data_path="run5/handeye_dataset.json",
        out_path="handeye_calib.json",
        setup="eye_to_hand"  # 或 "eye_in_hand" 或 "auto"
    )
    verify_handeye("run5/handeye_dataset.json", "handeye_calib.json")