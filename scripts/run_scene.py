from __future__ import annotations

import argparse
import pathlib
import time

import mujoco
import numpy as np
from mujoco.glfw import glfw

SCENE_PATH = pathlib.Path(__file__).resolve().parents[1] / "scene" / "task_scene.xml"
CAMERA_MAP = {
    "top": "top_cam",
    "grip": "grip_cam",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MuJoCo mini sorting scene viewer.")
    parser.add_argument(
        "--camera",
        choices=sorted(CAMERA_MAP.keys()),
        default="top",
        help="Initial main camera source. 'top' starts in free controllable mode.",
    )
    parser.add_argument(
        "--timestep",
        type=float,
        default=None,
        help="Override model timestep for debugging, e.g. 0.0015.",
    )
    return parser.parse_args()


def camera_id(model: mujoco.MjModel, camera_key: str) -> int:
    cam_name = CAMERA_MAP[camera_key]
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    if cam_id < 0:
        raise ValueError(f"Camera '{cam_name}' not found in model")
    return cam_id


def set_fixed_camera(cam: mujoco.MjvCamera, cam_id: int) -> None:
    cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    cam.fixedcamid = cam_id


def set_free_overview_camera(cam: mujoco.MjvCamera) -> None:
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = np.array([0.18, 0.0, 0.9])
    cam.distance = 1.45
    cam.azimuth = 145.0
    cam.elevation = -30.0


def set_joint_qpos(model: mujoco.MjModel, data: mujoco.MjData, joint_name: str, value: float) -> None:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        return
    qpos_adr = model.jnt_qposadr[joint_id]
    data.qpos[qpos_adr] = value


def set_initial_robot_pose(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    # Keep the arm folded above the basket side to avoid cabinet contact at startup.
    set_joint_qpos(model, data, "j1", -2.2)
    set_joint_qpos(model, data, "j2", 0.45)
    set_joint_qpos(model, data, "j3", -1.35)
    set_joint_qpos(model, data, "j4", 0.75)
    set_joint_qpos(model, data, "j5", 0.0)
    set_joint_qpos(model, data, "finger_left", -0.30)
    set_joint_qpos(model, data, "finger_right", -0.30)
    mujoco.mj_forward(model, data)



def apply_hold_control(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    # Hold joints near the initialized pose to avoid self-collision drift.
    qref = {
        "j1": -2.2,
        "j2": 0.45,
        "j3": -1.35,
        "j4": 0.75,
        "j5": 0.0,
    }

    kp = 28.0
    kd = 3.0
    arm_joint_order = ["j1", "j2", "j3", "j4", "j5"]

    for ctrl_id, joint_name in enumerate(arm_joint_order):
        if ctrl_id >= model.nu:
            break
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            continue
        qadr = model.jnt_qposadr[joint_id]
        dadr = model.jnt_dofadr[joint_id]
        pos_err = qref[joint_name] - data.qpos[qadr]
        vel = data.qvel[dadr]
        data.ctrl[ctrl_id] = kp * pos_err - kd * vel

    if model.nu >= 7:
        data.ctrl[5] = -0.30
        data.ctrl[6] = -0.30
def main() -> None:
    args = parse_args()

    if not SCENE_PATH.exists():
        raise FileNotFoundError(f"Scene file not found: {SCENE_PATH}")

    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)
    set_initial_robot_pose(model, data)

    if args.timestep is not None:
        if args.timestep <= 0:
            raise ValueError("--timestep must be positive")
        model.opt.timestep = args.timestep

    cam_ids = {key: camera_id(model, key) for key in CAMERA_MAP}

    main_key = args.camera
    inset_key = "grip" if main_key == "top" else "top"
    main_mode = "free" if main_key == "top" else "fixed"

    print("MuJoCo scene loaded")
    print(f"Scene: {SCENE_PATH}")
    print(f"Timestep: {model.opt.timestep}")
    print(f"Main source: {main_key} ({CAMERA_MAP[main_key]})")
    print(f"Inset source: {inset_key} ({CAMERA_MAP[inset_key]})")
    print("Controls:")
    print("  Mouse drag / wheel - rotate, pan, zoom main camera (in FREE mode)")
    print("  V - toggle main camera mode (FREE/FIXED)")
    print("  R - reset FREE camera to overview")
    print("  C - swap camera sources")
    print("  ESC or close window - exit")

    if not glfw.init():
        raise RuntimeError("Failed to initialize GLFW")

    width, height = 1400, 900
    window = glfw.create_window(width, height, "MuJoCo Mini Sorting Scene", None, None)
    if window is None:
        glfw.terminate()
        raise RuntimeError("Failed to create GLFW window")

    glfw.make_context_current(window)
    glfw.swap_interval(1)

    opt = mujoco.MjvOption()
    pert = mujoco.MjvPerturb()
    scn = mujoco.MjvScene(model, maxgeom=15000)
    con = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)

    main_cam = mujoco.MjvCamera()
    inset_cam = mujoco.MjvCamera()

    if main_mode == "free":
        set_free_overview_camera(main_cam)
    else:
        set_fixed_camera(main_cam, cam_ids[main_key])
    set_fixed_camera(inset_cam, cam_ids[inset_key])

    mouse_state = {
        "left": False,
        "middle": False,
        "right": False,
        "last_x": 0.0,
        "last_y": 0.0,
    }

    def key_callback(_window, key, _scancode, action, _mods) -> None:
        nonlocal main_key, inset_key, main_mode

        if action != glfw.PRESS:
            return

        if key == glfw.KEY_C:
            if main_mode == "fixed":
                main_key, inset_key = inset_key, main_key
                set_fixed_camera(main_cam, cam_ids[main_key])
                set_fixed_camera(inset_cam, cam_ids[inset_key])
                print(f"Swapped cameras -> main: {main_key}, inset: {inset_key}")
            else:
                inset_key = "grip" if inset_key == "top" else "top"
                set_fixed_camera(inset_cam, cam_ids[inset_key])
                print(f"Switched inset camera -> {inset_key}")
        elif key == glfw.KEY_V:
            if main_mode == "fixed":
                main_mode = "free"
                set_free_overview_camera(main_cam)
                print("Main camera mode: FREE")
            else:
                main_mode = "fixed"
                set_fixed_camera(main_cam, cam_ids[main_key])
                print(f"Main camera mode: FIXED ({main_key})")
        elif key == glfw.KEY_R and main_mode == "free":
            set_free_overview_camera(main_cam)
            print("Main free camera reset")

    def mouse_button_callback(_window, button, action, _mods) -> None:
        mouse_state["left"] = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        mouse_state["middle"] = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS
        mouse_state["right"] = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
        x, y = glfw.get_cursor_pos(window)
        mouse_state["last_x"] = x
        mouse_state["last_y"] = y

    def cursor_pos_callback(_window, xpos, ypos) -> None:
        if main_mode != "free":
            mouse_state["last_x"] = xpos
            mouse_state["last_y"] = ypos
            return

        dx = xpos - mouse_state["last_x"]
        dy = ypos - mouse_state["last_y"]
        mouse_state["last_x"] = xpos
        mouse_state["last_y"] = ypos

        if not (mouse_state["left"] or mouse_state["middle"] or mouse_state["right"]):
            return

        _, viewport_height = glfw.get_window_size(window)
        if viewport_height <= 0:
            return

        shift = (
            glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
            or glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
        )

        if mouse_state["right"]:
            action = mujoco.mjtMouse.mjMOUSE_MOVE_H if shift else mujoco.mjtMouse.mjMOUSE_MOVE_V
        elif mouse_state["left"]:
            action = mujoco.mjtMouse.mjMOUSE_ROTATE_H if shift else mujoco.mjtMouse.mjMOUSE_ROTATE_V
        else:
            action = mujoco.mjtMouse.mjMOUSE_ZOOM

        mujoco.mjv_moveCamera(model, action, dx / viewport_height, dy / viewport_height, scn, main_cam)

    def scroll_callback(_window, _xoffset, yoffset) -> None:
        if main_mode != "free":
            return
        mujoco.mjv_moveCamera(model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0.0, -0.08 * yoffset, scn, main_cam)

    glfw.set_key_callback(window, key_callback)
    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_cursor_pos_callback(window, cursor_pos_callback)
    glfw.set_scroll_callback(window, scroll_callback)

    try:
        while not glfw.window_should_close(window):
            step_start = time.time()
            t = data.time

            # Lightweight motion to demonstrate gripper camera follows end-effector.
            if model.nu >= 5:
                data.ctrl[0] = 0.35 * np.sin(0.8 * t)
                data.ctrl[1] = 0.28 * np.sin(0.7 * t + 0.7)
                data.ctrl[2] = 0.25 * np.sin(0.9 * t + 1.2)
                data.ctrl[3] = 0.22 * np.sin(1.0 * t + 1.8)
                data.ctrl[4] = 0.2 * np.sin(0.65 * t + 0.3)
            if model.nu >= 7:
                data.ctrl[5] = 0.0
                data.ctrl[6] = 0.0

            mujoco.mj_step(model, data)

            viewport_width, viewport_height = glfw.get_framebuffer_size(window)
            main_rect = mujoco.MjrRect(0, 0, viewport_width, viewport_height)

            mujoco.mjv_updateScene(
                model,
                data,
                opt,
                pert,
                main_cam,
                mujoco.mjtCatBit.mjCAT_ALL.value,
                scn,
            )
            mujoco.mjr_render(main_rect, scn, con)

            inset_width = max(260, viewport_width // 4)
            inset_height = max(180, viewport_height // 4)
            margin = 24
            inset_rect = mujoco.MjrRect(
                viewport_width - inset_width - margin,
                viewport_height - inset_height - margin,
                inset_width,
                inset_height,
            )

            mujoco.mjv_updateScene(
                model,
                data,
                opt,
                pert,
                inset_cam,
                mujoco.mjtCatBit.mjCAT_ALL.value,
                scn,
            )
            mujoco.mjr_render(inset_rect, scn, con)

            glfw.swap_buffers(window)
            glfw.poll_events()

            wait = model.opt.timestep - (time.time() - step_start)
            if wait > 0:
                time.sleep(wait)
    finally:
        glfw.destroy_window(window)
        glfw.terminate()


if __name__ == "__main__":
    main()















