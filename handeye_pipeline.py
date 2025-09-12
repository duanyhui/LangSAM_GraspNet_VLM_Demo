#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
handeye_pipeline.py  （修复 & 函数化版本，保持采集逻辑不变）

提供函数：
  - capture_handeye(...): 交互式采集，写入 {out}/handeye_dataset.json
  - solve_handeye(...):   调用 OpenCV Robot-World/Hand-Eye，并**按语义求逆后存储**外参
  - verify_handeye(...):  按 mode 自动选择等式，输出位置/角度误差

关键修复点：
  1) calibrateRobotWorldHandEye 的输出 ^wT_b、^cT_g 现在**按模式正确求逆**后保存为:
       - eye_to_hand:  T_base_cam = (^bT_c),  T_ee_board = (^eT_t)
       - eye_in_hand:  T_base_world = (^bT_w), T_gripper_cam = (^gT_c)
  2) 验证等式与模式严格一致
  3) Piper 姿态支持 rot_repr="euler" / "rodrigues"
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
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)  # 列优先；与 findChessboardCorners 返回顺序匹配
    objp[:,:2] = grid * (square_mm / 1000.0)
    return objp

def inv_T(T):
    """SE(3) 逆"""
    R = T[:3,:3]; t = T[:3,3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3,:3] = R.T
    Ti[:3,3]  = -R.T @ t
    return Ti

def se3_from_Rt(R, t):
    T = np.eye(4, dtype=np.float64)
    T[:3,:3] = R
    T[:3, 3] = t.reshape(3)
    return T

def rot_angle_deg_from_R(R):
    """旋转误差角（deg）"""
    tr = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return float(abs(np.degrees(np.arccos(tr))))

# ---------------- Piper 末端位姿读取 ----------------
def get_end_pose_data(piper, pos_unit_raw="mm", ang_unit_raw="deg"):
    """
    从 Piper SDK 读取 X,Y,Z,RX,RY,RZ（原始单位），统一换算到 mm/deg。
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

    if ang_unit_raw == "mdeg":
        rx_deg, ry_deg, rz_deg = rx_raw/1000.0, ry_raw/1000.0, rz_raw/1000.0
    else:
        rx_deg, ry_deg, rz_deg = float(rx_raw), float(ry_raw), float(rz_raw)

    return x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg

def get_ee_pose_from_sdk(piper: C_PiperInterface_V2,
                         pos_unit_raw="mm",
                         ang_unit_raw="deg",
                         euler_order="XYZ",
                         rot_repr="euler"):
    """
    返回 T_base_ee（单位 m）。支持两种姿态表示：
      - rot_repr="euler":   将 (RX,RY,RZ) 视为欧拉固定角，按 euler_order 组合
      - rot_repr="rodrigues": 将 (RX,RY,RZ) 视为“旋转向量”(角度制)；先转弧度再 cv2.Rodrigues
    注意：不同固件/SDK 版本语义可能不同。若验证出现 ~120-180° 的巨大角误差，请尝试 rot_repr="rodrigues"。
    """
    X,Y,Z,RX,RY,RZ = get_end_pose_data(piper, pos_unit_raw, ang_unit_raw)
    print("[DEBUG] 机械臂末端位姿（mm/deg）:",
          f"X={X:.1f}, Y={Y:.1f}, Z={Z:.1f}, RX={RX:.1f}, RY={RY:.1f}, RZ={RZ:.1f}")

    p = np.array([X, Y, Z], dtype=np.float64) / 1000.0

    if rot_repr.lower() == "rodrigues":
        # 把角度制的旋转向量转成弧度
        rotvec_rad = np.radians([RX, RY, RZ]).reshape(3,1)
        R, _ = cv2.Rodrigues(rotvec_rad)
    else:
        # 欧拉固定角（XYZ/ZYX）
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
            break

    if not ok:
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
                    rot_repr="euler",
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
        "samples": [],
        "meta": {
            "pos_unit_raw": pos_unit_raw,
            "ang_unit_raw": ang_unit_raw,
            "euler_order": euler_order,
            "rot_repr": rot_repr,
            "reproj_thresh": reproj_thresh
        }
    }

    print("\n[CAPTURE] 操作提示：")
    print("  - 移动机械臂让棋盘出现在不同位置/距离/姿态（覆盖视野四角、近中远、多角度）")
    print("  - 每到一个姿态，按 'SPACE' 采集；按 'q' 结束。建议 ≥ 20 组。")

    count = 0
    try:
        while True:
            try:
                time.sleep(0.01)
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
                cv2.putText(vis, "DETECTED", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,200,0), 2)
            else:
                cv2.putText(vis, "NOT FOUND", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)

            cv2.imshow("Capture (press SPACE to save, q to quit)", vis)
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            if key == 32 and ok:  # SPACE
                T_base_ee = get_ee_pose_from_sdk(
                    piper, pos_unit_raw, ang_unit_raw, euler_order, rot_repr
                )
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
def solve_handeye(data_path: str,
                  out_path: str = "./handeye_calib.json",
                  setup: str = "auto",
                  method: int = cv2.CALIB_ROBOT_WORLD_HAND_EYE_SHAH):
    """
    求解与保存标定：
      setup in {"auto","eye_to_hand","eye_in_hand"}
        - eye_to_hand：外置相机、棋盘固定在夹爪（想要 ^bT_c 与 ^eT_t）
        - eye_in_hand：相机在末端、棋盘固定在世界（想要 ^bT_w 与 ^gT_c）
      注意：按 OpenCV 语义，输出 ^wT_b 与 ^cT_g 需按模式**求逆后**再保存。
    """
    with open(data_path, "r", encoding="utf-8") as f:
        D = json.load(f)

    if setup == "auto":
        t_all = [np.array(s["T_cam_board"], dtype=np.float64)[:3,3] for s in D["samples"]]
        if len(t_all) < 3:
            setup = "eye_to_hand"
        else:
            std_meter = np.linalg.norm(np.std(np.stack(t_all, axis=0), axis=0))
            # 若相机相对棋盘的平移变化明显（>1cm），大概率是 eye_in_hand
            setup = "eye_in_hand" if std_meter > 0.01 else "eye_to_hand"
        print(f"[AUTO] 判定安装形态：{setup}")

    # 组装输入观测
    R_w2c, t_w2c, R_b2g, t_b2g = [], [], [], []
    for s in D["samples"]:
        T_cam_board = np.array(s["T_cam_board"], dtype=np.float64)  # ^cT_t
        T_base_ee   = np.array(s["T_base_ee"],   dtype=np.float64)  # ^bT_e
        if setup == "eye_to_hand":
            # world= camera, cam= board  → ^c' T_w' = ^board T camera = inv(^camera T board)
            T_w2c = inv_T(T_cam_board)
        elif setup == "eye_in_hand":
            # world= board, cam= camera  → ^c T_w = ^camera T board = ^cT_t
            T_w2c = T_cam_board
        else:
            raise ValueError("setup 必须是 {'auto','eye_to_hand','eye_in_hand'}")

        T_b2g = inv_T(T_base_ee)  # ^gT_b

        R_w2c.append(T_w2c[:3,:3]);      t_w2c.append(T_w2c[:3,3].reshape(3,1))
        R_b2g.append(T_b2g[:3,:3]);      t_b2g.append(T_b2g[:3,3].reshape(3,1))

    if not hasattr(cv2, "calibrateRobotWorldHandEye"):
        raise RuntimeError("当前 OpenCV 缺少 calibrateRobotWorldHandEye（需要 4.8+）。")

    R_base2world, t_base2world, R_gripper2cam, t_gripper2cam = cv2.calibrateRobotWorldHandEye(
        R_w2c, t_w2c, R_b2g, t_b2g, method=method
    )
    T_wTb = se3_from_Rt(R_base2world, t_base2world)  # ^wT_b
    T_cTg = se3_from_Rt(R_gripper2cam, t_gripper2cam) # ^cT_g （注意：若上面置换了 cam，这里的 c 就随之改变）

    # —— 关键：按模式把 OpenCV 输出“求逆后”保存成我们需要的量 —— #
    if setup == "eye_to_hand":
        # world'=camera, cam'=board
        T_base_cam  = inv_T(T_wTb)   # (^bT_c) = inv(^cT_b)
        T_ee_board  = inv_T(T_cTg)   # (^eT_t) = inv(^tT_e)
        out = {
            "mode": "eye_to_hand",
            "T_base_cam":  T_base_cam.tolist(),
            "T_ee_board":  T_ee_board.tolist(),
            "num_samples": len(D["samples"]),
            "method": int(method),
            "note": "world=camera, cam=board；已对 ^wT_b 与 ^c'T_g 求逆得到 ^bT_c / ^eT_t"
        }
    else:
        # eye_in_hand: world=board, cam=camera
        T_base_world = inv_T(T_wTb)  # (^bT_w) = inv(^wT_b)
        T_gripper_cam= inv_T(T_cTg)  # (^gT_c) = inv(^cT_g)
        out = {
            "mode": "eye_in_hand",
            "T_base_world":  T_base_world.tolist(),
            "T_gripper_cam": T_gripper_cam.tolist(),
            "num_samples": len(D["samples"]),
            "method": int(method),
            "note": "world=board, cam=camera；已对 ^wT_b 与 ^cT_g 求逆得到 ^bT_w / ^gT_c"
        }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[OK] 标定完成：写入 {out_path}  (mode={out['mode']})")
    return out_path

# ---------------- 验证 ----------------
def verify_handeye(data_path: str, calib_path: str):
    """
    根据 calib.json 中的 'mode' 字段自动选择校验公式。
    输出平均/最大 位置(单位 mm)/角度(单位 deg)误差。
    """
    with open(data_path, "r", encoding="utf-8") as f:
        D = json.load(f)
    with open(calib_path, "r", encoding="utf-8") as f:
        C = json.load(f)

    mode = C.get("mode", "eye_to_hand")
    pos_errs, ang_errs = [], []

    if mode == "eye_to_hand":
        T_base_cam = np.array(C["T_base_cam"], dtype=np.float64)   # ^bT_c
        T_ee_board = np.array(C["T_ee_board"], dtype=np.float64)   # ^eT_t

        for s in D["samples"]:
            T_cam_board = np.array(s["T_cam_board"], dtype=np.float64)  # ^cT_t
            T_base_ee   = np.array(s["T_base_ee"],   dtype=np.float64)  # ^bT_e

            # 预测 base<-board
            T_pred = T_base_cam @ T_cam_board                 # ^bT_c · ^cT_t = ^bT_t
            # “真值”
            T_gt   = T_base_ee @ T_ee_board                   # ^bT_e · ^eT_t = ^bT_t

            dp = T_pred[:3,3] - T_gt[:3,3]
            pos_errs.append(np.linalg.norm(dp) * 1000.0)

            dR = T_pred[:3,:3].T @ T_gt[:3,:3]
            ang_errs.append(rot_angle_deg_from_R(dR))

    elif mode == "eye_in_hand":
        T_base_world  = np.array(C["T_base_world"],  dtype=np.float64)  # ^bT_w
        T_gripper_cam = np.array(C["T_gripper_cam"], dtype=np.float64)  # ^gT_c

        for s in D["samples"]:
            T_cam_board = np.array(s["T_cam_board"], dtype=np.float64)  # ^cT_w (w=t=board)
            T_base_ee   = np.array(s["T_base_ee"],   dtype=np.float64)  # ^bT_e

            # 预测 base<-world
            T_pred = (T_base_ee @ T_gripper_cam) @ T_cam_board          # ^bT_e · ^gT_c · ^cT_w = ^bT_w
            T_gt   = T_base_world

            dp = T_pred[:3,3] - T_gt[:3,3]
            pos_errs.append(np.linalg.norm(dp) * 1000.0)

            dR = T_pred[:3,:3].T @ T_gt[:3,:3]
            ang_errs.append(rot_angle_deg_from_R(dR))
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
    # 示例：使用函数（不再依赖命令行）
    data_path = capture_handeye(
        out="run7",
        cols=7, rows=5, square_mm=25,
        pos_unit_raw="um", ang_unit_raw="mdeg",
        euler_order="XYZ",
        rot_repr="euler",          # 若误差依旧呈 100°+，改为 "rodrigues"
        reproj_thresh=0.4
    )
    calib_path = solve_handeye(
        data_path=data_path,
        out_path="handeye_calib.json",
        setup="eye_to_hand"        # 或 "eye_in_hand" 或 "auto"
    )
    verify_handeye(data_path, calib_path)
