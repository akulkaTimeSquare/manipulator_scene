from __future__ import annotations

import argparse
import json
import pathlib
import time
from dataclasses import dataclass, field

import mujoco
import numpy as np
from mujoco.glfw import glfw

SCENE_PATH = pathlib.Path(__file__).resolve().parents[1] / "scene" / "task_scene.xml"
CAMERA_MAP = {
    "top": "top_cam",
    "grip": "grip_cam",
}

ARM_JOINTS = ["j1", "j2", "j3", "j4", "j5"]
FINGER_JOINTS = ["finger_left", "finger_right"]


@dataclass
class EpisodeBuffer:
    actions: list[np.ndarray] = field(default_factory=list)
    qpos: list[np.ndarray] = field(default_factory=list)
    qvel: list[np.ndarray] = field(default_factory=list)
    ctrl: list[np.ndarray] = field(default_factory=list)
    sim_time: list[float] = field(default_factory=list)
    top_rgb: list[np.ndarray] = field(default_factory=list)
    grip_rgb: list[np.ndarray] = field(default_factory=list)

    def clear(self):
        self.actions.clear()
        self.qpos.clear()
        self.qvel.clear()
        self.ctrl.clear()
        self.sim_time.clear()
        self.top_rgb.clear()
        self.grip_rgb.clear()

    def size(self):
        return len(self.actions)


def parse_args():
    parser = argparse.ArgumentParser(description="Keyboard teleop for MuJoCo sorting scene with dataset logging.")
    parser.add_argument(
        "--camera",
        choices=sorted(CAMERA_MAP.keys()),
        default="top",
        help="Initial main camera source.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=pathlib.Path,
        default=pathlib.Path("data") / "teleop",
        help="Output directory for recorded episodes.",
    )
    parser.add_argument(
        "--record-hz",
        type=float,
        default=20.0,
        help="Sampling frequency used when recording trajectories.",
    )
    parser.add_argument(
        "--joint-speed",
        type=float,
        default=1.8,
        help="Joint target speed in rad/s.",
    )
    parser.add_argument(
        "--gripper-speed",
        type=float,
        default=1.5,
        help="Gripper command speed in joint units per second.",
    )
    parser.add_argument(
        "--save-rgb",
        action="store_true",
        help="Store rendered RGB frames for top/grip cameras in each episode file.",
    )
    return parser.parse_args()


def set_fixed_camera(cam, cam_id):
    cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    cam.fixedcamid = cam_id


def set_free_overview_camera(cam):
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = np.array([0.18, 0.0, 0.9])
    cam.distance = 1.45
    cam.azimuth = 145.0
    cam.elevation = -30.0


def camera_id(model, camera_key):
    cam_name = CAMERA_MAP[camera_key]
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    if cam_id < 0:
        raise ValueError(f"Camera '{cam_name}' not found")
    return cam_id


def set_joint_qpos(model, data, joint_name, value):
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        return
    qpos_adr = model.jnt_qposadr[joint_id]
    data.qpos[qpos_adr] = value


def set_initial_robot_pose(model, data):
    # Stable pose used by the viewer script to avoid startup self-collision drift.
    set_joint_qpos(model, data, "j1", -2.2)
    set_joint_qpos(model, data, "j2", 0.45)
    set_joint_qpos(model, data, "j3", -1.35)
    set_joint_qpos(model, data, "j4", 0.75)
    set_joint_qpos(model, data, "j5", 0.0)
    # Start from fully opened gripper.
    set_joint_qpos(model, data, "finger_left", -0.30)
    set_joint_qpos(model, data, "finger_right", -0.30)
    mujoco.mj_forward(model, data)


def resolve_joint_indices(model, joint_names):
    joint_ids: list[int] = []
    qpos_adr: list[int] = []
    dof_adr: list[int] = []
    for name in joint_names:
        j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if j_id < 0:
            raise ValueError(f"Joint '{name}' not found in model")
        joint_ids.append(j_id)
        qpos_adr.append(int(model.jnt_qposadr[j_id]))
        dof_adr.append(int(model.jnt_dofadr[j_id]))
    return np.array(joint_ids), np.array(qpos_adr), np.array(dof_adr)


def joint_limits_for(model, joint_ids):
    limits = np.zeros((len(joint_ids), 2), dtype=np.float64)
    for i, j_id in enumerate(joint_ids):
        limits[i, 0] = float(model.jnt_range[j_id, 0])
        limits[i, 1] = float(model.jnt_range[j_id, 1])
    return limits


def clamp(values, limits):
    return np.minimum(np.maximum(values, limits[:, 0]), limits[:, 1])


def next_episode_index(dataset_dir):
    max_idx = -1
    for npz_path in dataset_dir.glob("episode_*.npz"):
        stem = npz_path.stem
        suffix = stem.replace("episode_", "", 1)
        if suffix.isdigit():
            max_idx = max(max_idx, int(suffix))
    return max_idx + 1


def write_index_line(dataset_dir, metadata):
    index_path = dataset_dir / "index.jsonl"
    with index_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(metadata, ensure_ascii=True) + "\n")


def render_rgb(renderer, data, cam_name):
    renderer.update_scene(data, camera=cam_name)
    image = renderer.render()
    return image.copy()


def main():
    args = parse_args()

    if args.record_hz <= 0:
        raise ValueError("--record-hz must be positive")
    if args.joint_speed <= 0:
        raise ValueError("--joint-speed must be positive")
    if args.gripper_speed <= 0:
        raise ValueError("--gripper-speed must be positive")
    if not SCENE_PATH.exists():
        raise FileNotFoundError(f"Scene file not found: {SCENE_PATH}")

    args.dataset_dir.mkdir(parents=True, exist_ok=True)
    episode_idx = next_episode_index(args.dataset_dir)

    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)
    set_initial_robot_pose(model, data)
    initial_qpos = data.qpos.copy()
    initial_qvel = data.qvel.copy()

    arm_joint_ids, arm_qadr, arm_dadr = resolve_joint_indices(model, ARM_JOINTS)
    finger_joint_ids, finger_qadr, _ = resolve_joint_indices(model, FINGER_JOINTS)
    arm_limits = joint_limits_for(model, arm_joint_ids)
    finger_limits = joint_limits_for(model, finger_joint_ids)

    target_arm_q = data.qpos[arm_qadr].copy()
    target_grip = float(data.qpos[finger_qadr[0]])
    target_grip = float(np.clip(target_grip, finger_limits[0, 0], finger_limits[0, 1]))

    arm_ctrl_count = min(len(ARM_JOINTS), model.nu)
    if model.nu < 7:
        raise RuntimeError("Expected at least 7 actuators (5 arm + 2 gripper)")

    cam_ids = {key: camera_id(model, key) for key in CAMERA_MAP}
    main_key = args.camera
    inset_key = "grip" if main_key == "top" else "top"
    main_mode = "fixed"

    if not glfw.init():
        raise RuntimeError("Failed to initialize GLFW")

    width, height = 1400, 900
    window = glfw.create_window(width, height, "MuJoCo Teleop Collector", None, None)
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
    set_fixed_camera(main_cam, cam_ids[main_key])
    set_fixed_camera(inset_cam, cam_ids[inset_key])

    key_down: dict[int, bool] = {}
    mouse_state = {
        "left": False,
        "middle": False,
        "right": False,
        "last_x": 0.0,
        "last_y": 0.0,
    }

    episode = EpisodeBuffer()
    recording = False
    success_label = False
    record_period = 1.0 / args.record_hz
    last_record_time = 0.0
    top_renderer = mujoco.Renderer(model, width=320, height=240) if args.save_rgb else None
    grip_renderer = mujoco.Renderer(model, width=320, height=240) if args.save_rgb else None

    joint_key_pairs = [
        (glfw.KEY_1, glfw.KEY_Q),
        (glfw.KEY_2, glfw.KEY_W),
        (glfw.KEY_3, glfw.KEY_E),
        (glfw.KEY_4, glfw.KEY_R),
        (glfw.KEY_5, glfw.KEY_T),
    ]

    def save_episode():
        nonlocal episode_idx
        if episode.size() == 0:
            print("Episode is empty, nothing saved.")
            return

        npz_name = f"episode_{episode_idx:05d}.npz"
        json_name = f"episode_{episode_idx:05d}.json"
        npz_path = args.dataset_dir / npz_name
        json_path = args.dataset_dir / json_name

        payload = {
            "actions": np.asarray(episode.actions, dtype=np.float32),
            "qpos": np.asarray(episode.qpos, dtype=np.float32),
            "qvel": np.asarray(episode.qvel, dtype=np.float32),
            "ctrl": np.asarray(episode.ctrl, dtype=np.float32),
            "time": np.asarray(episode.sim_time, dtype=np.float64),
        }
        if args.save_rgb:
            payload["top_rgb"] = np.asarray(episode.top_rgb, dtype=np.uint8)
            payload["grip_rgb"] = np.asarray(episode.grip_rgb, dtype=np.uint8)

        np.savez_compressed(npz_path, **payload)

        metadata = {
            "episode_index": episode_idx,
            "steps": episode.size(),
            "success": bool(success_label),
            "npz_file": npz_name,
            "timestamp_unix": time.time(),
            "record_hz": args.record_hz,
            "save_rgb": bool(args.save_rgb),
        }

        with json_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=True)
        write_index_line(args.dataset_dir, metadata)
        print(f"Saved episode {episode_idx:05d}: {episode.size()} steps, success={success_label}")
        episode_idx += 1

    def reset_scene():
        nonlocal target_grip
        data.qpos[:] = initial_qpos
        data.qvel[:] = initial_qvel
        data.ctrl[:] = 0
        mujoco.mj_forward(model, data)
        target_arm_q[:] = data.qpos[arm_qadr]
        target_grip = float(np.clip(data.qpos[finger_qadr[0]], finger_limits[0, 0], finger_limits[0, 1]))
        print("Scene reset.")

    def stop_recording(save):
        nonlocal recording
        if not recording:
            return
        recording = False
        if save:
            save_episode()
        else:
            print("Recording discarded.")
        episode.clear()

    def start_recording():
        nonlocal recording, success_label, last_record_time
        if recording:
            return
        episode.clear()
        recording = True
        success_label = False
        last_record_time = data.time - record_period
        print("Recording started.")

    def key_callback(_window, key, _scancode, action, _mods):
        nonlocal main_key, inset_key, main_mode, success_label, target_grip

        if action == glfw.PRESS:
            key_down[key] = True
        elif action == glfw.RELEASE:
            key_down[key] = False

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
        elif key == glfw.KEY_ENTER:
            if recording:
                stop_recording(save=True)
            else:
                start_recording()
        elif key == glfw.KEY_BACKSPACE:
            stop_recording(save=False)
        elif key == glfw.KEY_N:
            stop_recording(save=False)
            reset_scene()
        elif key == glfw.KEY_Y:
            success_label = True
            print("Marked current episode as SUCCESS.")
        elif key == glfw.KEY_U:
            success_label = False
            print("Marked current episode as FAILURE.")
        elif key == glfw.KEY_Z:
            target_grip = float(finger_limits[0, 1])
            print("Gripper target: fully OPEN")
        elif key == glfw.KEY_X:
            target_grip = float(finger_limits[0, 0])
            print("Gripper target: fully CLOSE")

    def mouse_button_callback(_window, _button, _action, _mods):
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

    print("MuJoCo teleop collector loaded")
    print(f"Scene: {SCENE_PATH}")
    print(f"Dataset dir: {args.dataset_dir.resolve()}")
    print("Controls:")
    print("  Arm joints: 1/Q, 2/W, 3/E, 4/R, 5/T")
    print("  Wrist roll (j5) alternative: [ (negative), ] (positive)")
    print("    (number key -> positive, letter key -> negative)")
    print("  Gripper hold: O (open), P (close)")
    print("  Gripper snap: Z (fully open), X (fully close)")
    print("  ENTER: start/stop recording (stop saves)")
    print("  BACKSPACE: discard current recording")
    print("  Y/U: mark success/failure label")
    print("  N: reset scene and discard current recording")
    print("  C/V/R + mouse: camera controls")
    print("  ESC: exit")

    reset_scene()

    try:
        while not glfw.window_should_close(window):
            step_start = time.time()
            dt = model.opt.timestep

            action_cmd = np.zeros(6, dtype=np.float32)
            for i, (k_pos, k_neg) in enumerate(joint_key_pairs):
                direction = float(key_down.get(k_pos, False)) - float(key_down.get(k_neg, False))
                action_cmd[i] = direction
                target_arm_q[i] += direction * args.joint_speed * dt
            target_arm_q[:] = clamp(target_arm_q, arm_limits)

            open_cmd = float(key_down.get(glfw.KEY_O, False))
            close_cmd = float(key_down.get(glfw.KEY_P, False))
            grip_direction = close_cmd - open_cmd
            action_cmd[5] = grip_direction
            target_grip += grip_direction * args.gripper_speed * dt
            target_grip = float(np.clip(target_grip, finger_limits[0, 0], finger_limits[0, 1]))

            kp = 28.0
            kd = 3.0
            for ctrl_id in range(arm_ctrl_count):
                q = data.qpos[arm_qadr[ctrl_id]]
                dq = data.qvel[arm_dadr[ctrl_id]]
                data.ctrl[ctrl_id] = kp * (target_arm_q[ctrl_id] - q) - kd * dq

            data.ctrl[5] = target_grip
            data.ctrl[6] = target_grip

            mujoco.mj_step(model, data)

            if recording and (data.time - last_record_time >= record_period):
                episode.actions.append(action_cmd.copy())
                episode.qpos.append(data.qpos.copy())
                episode.qvel.append(data.qvel.copy())
                episode.ctrl.append(data.ctrl.copy())
                episode.sim_time.append(float(data.time))
                if args.save_rgb:
                    if top_renderer is None or grip_renderer is None:
                        raise RuntimeError("Renderers are not initialized")
                    episode.top_rgb.append(render_rgb(top_renderer, data, CAMERA_MAP["top"]))
                    episode.grip_rgb.append(render_rgb(grip_renderer, data, CAMERA_MAP["grip"]))
                last_record_time = data.time

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
        if top_renderer is not None:
            top_renderer.close()
        if grip_renderer is not None:
            grip_renderer.close()
        glfw.destroy_window(window)
        glfw.terminate()


if __name__ == "__main__":
    main()


























