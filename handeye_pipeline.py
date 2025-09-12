#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
handeye_pipeline.py
功能：
  - capture：采集多组样本（相机检测棋盘 + 读取机械臂 EE 位姿），保存数据集
  - solve  ：使用 OpenCV 进行 Robot-World/Hand-Eye 联合标定，输出 T_base_cam 与 T_ee_board
  - verify ：用求解结果在采样数据上做几何一致性检查（位置/角度误差）

依赖：
  pip install opencv-python numpy

注意：
  - 复用你的 orbbec_io.py（OrbbecCamera 类）来抓取彩色图像与导出相机内参
  - 复用你的 Piper SDK（GetArmEndPoseMsgs / move_to_pose）
"""

from __future__ import annotations
import os, json, time, argparse
import numpy as np
import cv2

# === 引用你现有的工具 ===
from orbbec_io import OrbbecCamera      # 相机采集/导出内参（来自 orbbec_io.py）  # ← 参见你提供的文件
from piper_sdk import C_PiperInterface_V2
# 读取机械臂末端位姿：参考 piper_read_end_pose.py 中的调用方式（GetArmEndPoseMsgs）
# 实际返回结构可能因 SDK 版本而异，请在 get_ee_pose_from_sdk() 中按需要微调。
# 运动下发会在应用抓取阶段演示（可与 piper_method.py 的 move_to_pose 结合）

# ---------- 基础数学 ----------
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

def euler_xyz_from_R(R):
    """R -> (rx,ry,rz) in degrees, XYZ 固定角"""
    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
    if sy > 1e-9:
        rx = np.arctan2(R[2,1], R[2,2])
        ry = np.arctan2(-R[2,0], sy)
        rz = np.arctan2(R[1,0], R[0,0])
    else:
        rx = np.arctan2(-R[1,2], R[1,1]); ry = np.arctan2(-R[2,0], sy); rz = 0.0
    return np.degrees([rx, ry, rz])

def make_chessboard_object_points(cols, rows, square_mm):
    """构造棋盘格内角点三维坐标（单位：m；z=0 平面）"""
    objp = np.zeros((rows*cols, 3), np.float32)
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)  # 注意：列在前、行在后
    objp[:,:2] = grid * (square_mm / 1000.0)          # 转米
    return objp  # (N,3) in meters


def get_end_pose_data(piper, pos_unit_raw="mm", ang_unit_raw="deg"):
    """
    仅获取机械臂末端位姿数据，不打印

    Args:
        piper: C_PiperInterface_V2对象
        pos_unit_raw: 机械臂回报的“原始位置单位”（"mm" 或 "um"）
        ang_unit_raw: 机械臂回报的“原始角度单位”（"deg" 或 "mdeg"）

    Returns:
        tuple: (x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg) 六个位姿数据（统一换算为 mm / deg）
    """
    end_pose_msg = piper.GetArmEndPoseMsgs()

    # 原始单位 → 统一为 mm / deg（根据控制器真实回报单位切换）
    x_raw = end_pose_msg.end_pose.X_axis
    y_raw = end_pose_msg.end_pose.Y_axis
    z_raw = end_pose_msg.end_pose.Z_axis
    rx_raw = end_pose_msg.end_pose.RX_axis
    ry_raw = end_pose_msg.end_pose.RY_axis
    rz_raw = end_pose_msg.end_pose.RZ_axis

    # 位置：支持 "mm" 或 "um"
    x_mm = (x_raw / 1000.0) if pos_unit_raw == "um" else float(x_raw)
    y_mm = (y_raw / 1000.0) if pos_unit_raw == "um" else float(y_raw)
    z_mm = (z_raw / 1000.0) if pos_unit_raw == "um" else float(z_raw)

    # 角度：支持 "deg" 或 "mdeg"
    rx_deg = (rx_raw / 1000.0) if ang_unit_raw == "mdeg" else float(rx_raw)
    ry_deg = (ry_raw / 1000.0) if ang_unit_raw == "mdeg" else float(ry_raw)
    rz_deg = (rz_raw / 1000.0) if ang_unit_raw == "mdeg" else float(rz_raw)

    return x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg

# ---------- 机械臂位姿读取（需按你的 SDK 返回进行适配） ----------
def get_ee_pose_from_sdk(piper: C_PiperInterface_V2,
                         pos_unit_raw="mm", ang_unit_raw="deg", euler_order="XYZ"):
    """
    返回 T_base_ee 齐次矩阵（单位：米；姿态由度→旋转矩阵）。
    参考：piper_read_end_pose.py 的示例用法（GetArmEndPoseMsgs）。你需要根据实际返回值结构适配解析。
    euler_order: 控制器欧拉角顺序，支持 "XYZ" / "ZYX"
    """
    msg = piper.GetArmEndPoseMsgs()  # 例如可能返回 dict 或列表/元组
    # 解析 msg，提取位姿数据（单位：mm / deg）
    X,Y,Z,RX,RY,RZ = get_end_pose_data(piper, pos_unit_raw, ang_unit_raw)
    print("[DEBUG] 机械臂末端位姿（mm/deg）:", f"X={X:.1f}, Y={Y:.1f}, Z={Z:.1f}, RX={RX:.1f}, RY={RY:.1f}, RZ={RZ:.1f}")

    # 单位换算：mm -> m；deg -> rad -> R
    p = np.array([X, Y, Z], dtype=np.float64) / 1000.0
    r = np.radians([RX, RY, RZ])

    # 欧拉固定角（与控制器欧拉约定需一致）：支持 XYZ / ZYX
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
    else:  # "ZYX"
        R = Rz @ Ry @ Rx

    T = np.eye(4); T[:3,:3] = R; T[:3,3] = p
    return T

# ---------- 相机棋盘姿态估计 ----------
def detect_chessboard_pose(color_bgr, cols, rows, square_mm, K, dist, reproj_thresh=0.4):
    """返回 (ok, T_cam_board)。失败返回 (False, None)。单位：米。"""
    gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
    pattern_size = (cols, rows)  # 列×行（内角点数）

    # 尝试多种检测标志组合
    flag_combinations = [
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FILTER_QUADS,
        cv2.CALIB_CB_ADAPTIVE_THRESH,
        cv2.CALIB_CB_NORMALIZE_IMAGE,
        0  # 无标志
    ]

    ok = False
    corners = None
    for flags in flag_combinations:
        ok, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
        if ok:
            print(f"[DEBUG] 棋盘格检测成功，使用标志: {flags}, 检测到 {len(corners)} 个角点")
            break

    if not ok:
        print(f"[DEBUG] 棋盘格检测失败，尝试了 {len(flag_combinations)} 种方法")
        print(f"[DEBUG] 期望检测: {cols}x{rows} = {cols*rows} 个内角点")
        return False, None

    # 亚像素细化
    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), term)

    # 构造棋盘 3D 点（Z=0 平面，单位米）
    objp = make_chessboard_object_points(cols, rows, square_mm)

    # PnP 解姿态（迭代法）
    ok, rvec, tvec = cv2.solvePnP(objp, corners, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        print("[DEBUG] PnP 求解失败")
        return False, None

    proj, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
    reproj = np.mean(np.linalg.norm(proj.squeeze() - corners.squeeze(), axis=1))
    if reproj > reproj_thresh:
        print(f"[DROP] reproj={reproj:.3f}px");
        return False, None
    T_cam_board = rvec_tvec_to_T(rvec, tvec)
    print(f"[DEBUG] 棋盘格位姿求解成功，距离: {np.linalg.norm(tvec):.3f}m")
    return True, T_cam_board

# ---------- 数据采集 ----------
def run_capture(args):
    os.makedirs(args.out, exist_ok=True)
    # 1) 相机上电
    cam = OrbbecCamera(width=args.width, height=args.height, fps=args.fps)
    cam.start()
    print(f"[INFO] Orbbec align mode: {cam.align_mode}")

    # 导出一次 intrinsics.json（含 fx,fy,cx,cy, coeffs 等）
    intr_path = os.path.join(args.out, "intrinsics.json")
    # 抓一帧用于获取深度尺度等（不强制用深度，仅为导出完整信息）
    try:
        cf, df = cam.capture(timeout_ms=6000)
        cam.export_intrinsics_json(intr_path, depth_frame=df)
    except RuntimeError as e:
        print(f"[WARNING] 初始采集失败，可能深度帧缺失: {e}")
        # 重试几次或使用备用方案
        for retry in range(3):
            try:
                cf, df = cam.capture(timeout_ms=3000)
                cam.export_intrinsics_json(intr_path, depth_frame=df)
                break
            except RuntimeError:
                if retry == 2:
                    print("[WARNING] 多次重试失败，使用仅彩色内参导出")
                    cf, _ = cam.capture(timeout_ms=3000)  # 忽略深度帧
                    cam.export_intrinsics_json(intr_path, depth_frame=None)
                time.sleep(0.5)

    # 直接从 profile 读取内参
    c_intr = cam.color_profile.get_intrinsic()
    fx, fy, cx, cy = c_intr.fx, c_intr.fy, c_intr.cx, c_intr.cy
    print("fx, fy, cx, cy =", fx, fy, cx, cy)

    try:
        dist_coeffs = np.array(list(c_intr.coeffs), dtype=np.float64).reshape(-1,1)
    except Exception:
        dist_coeffs = np.zeros((5,1), dtype=np.float64)  # 若 SDK 不提供，兜底为零畸变
    K = np.array([[fx, 0, cx],[0, fy, cy],[0, 0, 1]], dtype=np.float64)

    # 2) 机械臂上电
    piper = C_PiperInterface_V2("can0")
    piper.ConnectPort()
    while not piper.EnablePiper():
        time.sleep(0.01)
    print("[INFO] Piper connected & enabled.")

    # 3) 采集循环
    dataset = {
        "cols": args.cols, "rows": args.rows, "square_mm": args.square_mm,
        "K": K.tolist(), "dist": dist_coeffs.reshape(-1).tolist(),
        "samples": []
    }
    print("\n[CAPTURE] 操作提示：")
    print("  - 移动机械臂让棋盘出现在不同位置/距离/姿态（覆盖视野四角、近中远、多角度）")
    print("  - 每到一个姿态，按 'SPACE' 采集；按 'q' 结束。建议 ≥ 20 组。")

    count = 0
    while True:
        try:
            time.sleep(0.1)
            cf, df = cam.capture(timeout_ms=3000)  # 缩短超时时间
            color_bgr, _ = cam.frames_to_numpy(cf, df)
        except RuntimeError as e:
            print(f"[WARNING] 采集失败: {e}, 跳过此帧")
            continue

        vis = color_bgr.copy()
        ok, T_cam_board = detect_chessboard_pose(
            color_bgr, args.cols, args.rows, args.square_mm, K, dist_coeffs, args.reproj_thresh
        )
        if ok:
            # 可视化角点
            pattern_size = (args.cols, args.rows)
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
            # 读取 EE 位姿（根据原始单位与欧拉顺序转换）
            T_base_ee = get_ee_pose_from_sdk(piper, args.pos_unit_raw, args.ang_unit_raw, args.euler_order)
            # 保存样本
            sample = {
                "T_cam_board": T_cam_board.tolist(),
                "T_base_ee": T_base_ee.tolist(),
                "snapshot": f"cap_{count:03d}.png"
            }
            dataset["samples"].append(sample)
            cv2.imwrite(os.path.join(args.out, sample["snapshot"]), color_bgr)
            count += 1
            print(f"[SAVED] sample #{count}")

    cv2.destroyAllWindows()
    # 落库
    data_path = os.path.join(args.out, "handeye_dataset.json")
    with open(data_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    print(f"\n[DONE] 共采集 {len(dataset['samples'])} 组样本 → {data_path}")

    # 清理资源
    cam.stop()

# ---------- 求解 ----------
def run_solve(args):
    with open(args.data, "r", encoding="utf-8") as f:
        D = json.load(f)
    K = np.array(D["K"], dtype=np.float64)
    dist = np.array(D["dist"], dtype=np.float64).reshape(-1,1)

    # 构造绝对位姿序列
    R_gripper2base, t_gripper2base = [], []
    R_target2cam,  t_target2cam  = [], []
    for s in D["samples"]:
        T_cam_board = np.array(s["T_cam_board"], dtype=np.float64)
        # 可选：若数据中保存的是 T_board_cam，这里取逆
        if args.invert_cam_board:
            T_cam_board = np.linalg.inv(T_cam_board)
        T_base_ee   = np.array(s["T_base_ee"],   dtype=np.float64)
        R_target2cam.append(T_cam_board[:3,:3])
        t_target2cam.append(T_cam_board[:3, 3].reshape(3,1))
        R_gripper2base.append(T_base_ee[:3,:3])   # “末端到基座”的绝对姿态
        t_gripper2base.append(T_base_ee[:3, 3].reshape(3,1))

    # 优先使用 OpenCV 的联合标定（需要较新的 OpenCV）
    if hasattr(cv2, "calibrateRobotWorldHandEye"):
        R_base_cam, t_base_cam, R_ee_board, t_ee_board = cv2.calibrateRobotWorldHandEye(
            R_gripper2base, t_gripper2base,
            R_target2cam,  t_target2cam
        )
    else:
        raise RuntimeError("当前 OpenCV 缺少 calibrateRobotWorldHandEye，请升级到较新版本（4.8+）。")

    # 打包并保存
    T_base_cam = np.eye(4); T_base_cam[:3,:3]=R_base_cam; T_base_cam[:3,3]=t_base_cam.reshape(3)
    T_ee_board = np.eye(4); T_ee_board[:3,:3]=R_ee_board; T_ee_board[:3,3]=t_ee_board.reshape(3)

    out = {
        "T_base_cam": T_base_cam.tolist(),
        "T_ee_board": T_ee_board.tolist(),
        "num_samples": len(D["samples"])
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[OK] 标定完成：写入 {args.out}")

# ---------- 验证 ----------
def run_verify(args):
    with open(args.data, "r", encoding="utf-8") as f:
        D = json.load(f)
    with open(args.calib, "r", encoding="utf-8") as f:
        C = json.load(f)

    T_base_cam = np.array(C["T_base_cam"], dtype=np.float64)
    T_ee_board = np.array(C["T_ee_board"], dtype=np.float64)

    pos_errs, ang_errs = [], []
    for s in D["samples"]:
        T_cam_board = np.array(s["T_cam_board"], dtype=np.float64)
        if args.invert_cam_board:
            T_cam_board = np.linalg.inv(T_cam_board)
        T_base_ee   = np.array(s["T_base_ee"],   dtype=np.float64)

        # 用解出的外参把“相机看到的板”变到基座系：
        T_base_board_pred = T_base_cam @ T_cam_board
        # 而真实的“板在基座系”应为：T_base_ee @ T_ee_board
        T_base_board_gt   = T_base_ee @ T_ee_board

        dp = T_base_board_pred[:3,3] - T_base_board_gt[:3,3]
        pos_errs.append(np.linalg.norm(dp)*1000.0)  # m -> mm

        R_pred = T_base_board_pred[:3,:3]
        R_gt   = T_base_board_gt[:3,:3]
        dR = R_pred.T @ R_gt
        angle = np.degrees(np.arccos(np.clip((np.trace(dR)-1)/2, -1, 1)))
        ang_errs.append(abs(angle))

    print(f"[VERIFY] 平均位置误差: {np.mean(pos_errs):.2f} mm  | 最大: {np.max(pos_errs):.2f} mm")
    print(f"[VERIFY] 平均角度误差: {np.mean(ang_errs):.3f} deg | 最大: {np.max(ang_errs):.3f} deg")

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Eye-to-Hand 手眼标定流水线")
    sub = ap.add_subparsers(dest="cmd")

    ap_cap = sub.add_parser("capture", help="采集数据样本")
    ap_cap.add_argument("--out", type=str, default="./handeye_run", help="输出目录")
    ap_cap.add_argument("--cols", type=int, default=5, help="棋盘内角点 列数（例如 10）")
    ap_cap.add_argument("--rows", type=int, default=7, help="棋盘内角点 行数（例如 7）")
    ap_cap.add_argument("--square-mm", dest="square_mm", type=float, default=25.0, help="每格边长（mm）")
    ap_cap.add_argument("--width", type=int, default=1280)
    ap_cap.add_argument("--height", type=int, default=800)
    ap_cap.add_argument("--fps", type=int, default=30)
    # 新增：单位/欧拉/重投影阈值
    ap_cap.add_argument("--pos-unit-raw", choices=["mm","um"], default="mm",
                        help="机械臂回报的位置原始单位（mm 或 um）")
    ap_cap.add_argument("--ang-unit-raw", choices=["deg","mdeg"], default="deg",
                        help="机械臂回报的角度原始单位（deg 或 mdeg）")
    ap_cap.add_argument("--euler-order", choices=["XYZ","ZYX"], default="XYZ",
                        help="控制器的欧拉角顺序（固定角），常见 XYZ 或 ZYX")
    ap_cap.add_argument("--reproj-thresh", type=float, default=0.4,
                        help="采样阶段的重投影误差阈值（像素），超过将丢弃该样本")
    ap_cap.set_defaults(func=run_capture)

    ap_sol = sub.add_parser("solve", help="求解手眼外参")
    ap_sol.add_argument("--data", type=str, default="./handeye_dataset.json", help="handeye_dataset.json 路径")
    ap_sol.add_argument("--out",  type=str, default="./handeye_calib.json", help="输出外参 JSON 路径")
    ap_sol.add_argument("--invert-cam-board", action="store_true",
                        help="若数据中保存的是 T_board_cam，则开启该开关在求解/验证时取逆")
    ap_sol.set_defaults(func=run_solve)

    ap_ver = sub.add_parser("verify", help="验证求解结果")
    ap_ver.add_argument("--data",  type=str, default="./handeye_dataset.json", help="handeye_dataset.json 路径")
    ap_ver.add_argument("--calib", type=str, default="./handeye_calib.json", help="handeye_calib.json 路径")
    ap_ver.add_argument("--invert-cam-board", action="store_true",
                        help="与 --invert-cam-board 一致，验证阶段是否对 T_cam_board 取逆")
    ap_ver.set_defaults(func=run_verify)

    args = ap.parse_args()
    # ---- 修复: 未提供子命令时给出帮助而不是 AttributeError ----
    if not hasattr(args, 'func'):
        ap.print_help()
        print("\n示例: \n  python handeye_pipeline.py capture --out ./run1"\
              "\n  python handeye_pipeline.py solve --data ./run1/handeye_dataset.json --out ./handeye_calib.json"\
              "\n  python handeye_pipeline.py verify --data ./run1/handeye_dataset.json --calib ./handeye_calib.json")
        return
    args.func(args)

if __name__ == "__main__":
    main()
