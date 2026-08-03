"""
Visualization utilities for MANO hand mesh overlay on Hot3D Aria RGB frames.

Uses the same frame-to-timecode mapping and projection logic as the standalone
visualize_mano_hands.py script to ensure correct alignment.
"""

import bisect
import csv
import json
import os

import cv2
import numpy as np
import smplx
import torch
from decord import VideoReader
# bypassed global import
# bypassed global import
from scipy.spatial.transform import Rotation


# ---------------------------------------------------------------------------
# MANO mesh generation
# ---------------------------------------------------------------------------

def _quat_wxyz_to_rotvec(q_wxyz, eps=1e-8):
    """Convert a (w, x, y, z) quaternion to a rotation vector.

    Predicted quaternions early in training can have vanishing norm, which
    crashes scipy's `Rotation.from_quat`. Normalise and fall back to identity
    when the norm is below `eps`.
    """
    q = np.asarray(q_wxyz, dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm < eps:
        q = np.array([1.0, 0.0, 0.0, 0.0])
    else:
        q = q / norm
    q_xyzw = np.array([q[1], q[2], q[3], q[0]])
    return Rotation.from_quat(q_xyzw).as_rotvec().astype(np.float32)


def quat_wxyz_to_axis_angle_torch(q_wxyz, eps=1e-8):
    """Differentiable (w, x, y, z) quaternion -> axis-angle vector.

    Args:
        q_wxyz (torch.Tensor): [..., 4]
    Returns:
        torch.Tensor: [..., 3] rotation vector such that R = exp([v]_x).

    Degenerate (near-zero-norm) quaternions — absent-hand fillers or a collapsed
    head prediction — are replaced with identity BEFORE the conversion. Without
    this guard their axis-angle Jacobian is ~pi/eps^2 (~3e16): finite in the
    forward but an enormous gradient that becomes NaN once it propagates and
    combines under reduced precision. That NaN gradient froze P1a training
    (job 97944) — the loss stayed finite while grad_norm went NaN and the
    NaN-guard skipped every optimizer step. ``torch.where`` zeroes the gradient
    for the replaced entries. Present near-identity rotations (norm ~1, tiny
    vector part) are NOT degenerate and pass through with a finite gradient.
    """
    identity = torch.zeros_like(q_wxyz)
    identity[..., 0] = 1.0
    q_wxyz = torch.where(q_wxyz.norm(dim=-1, keepdim=True) < 1e-6, identity, q_wxyz)

    q = q_wxyz / q_wxyz.norm(dim=-1, keepdim=True).clamp_min(eps)
    w = q[..., :1]
    xyz = q[..., 1:]
    # eps INSIDE the sqrt (not clamp_min after norm): keeps the backward of
    # sin_half finite as a real rotation passes through identity (vector -> 0).
    sin_half = torch.sqrt((xyz * xyz).sum(dim=-1, keepdim=True) + eps * eps)
    angle = 2.0 * torch.atan2(sin_half, w.abs())
    sign = torch.where(w >= 0, torch.ones_like(w), -torch.ones_like(w))
    axis = sign * xyz / sin_half
    return angle * axis


class MANOModel:
    """Wrapper around smplx MANO for left/right hand mesh generation."""

    def __init__(self, mano_model_folder):
        self.left = smplx.create(
            os.path.join(mano_model_folder, "MANO_LEFT.pkl"),
            "mano",
            use_pca=True,
            is_rhand=False,
            num_pca_comps=15,
        )
        self.right = smplx.create(
            os.path.join(mano_model_folder, "MANO_RIGHT.pkl"),
            "mano",
            use_pca=True,
            is_rhand=True,
            num_pca_comps=15,
        )
        # Fix left hand shapedirs bug (https://github.com/vchoutas/smplx/issues/48)
        if (
            torch.sum(
                torch.abs(self.left.shapedirs[:, 0, :] - self.right.shapedirs[:, 0, :])
            )
            < 1
        ):
            self.left.shapedirs[:, 0, :] *= -1

        # HOI4D / manopth convention: the released poseCoeff carries a FULL 45-dim
        # axis-angle hand pose with flat_hand_mean=True (no mean folded in). The PCA(15)
        # + smplx-default flat_hand_mean=False layers above both truncate that pose AND
        # add ~11.7rad of hand-mean bend, which scrambles every finger joint. These
        # dedicated layers consume the raw 45-dim pose exactly as HOI4D intends, so MANO
        # forward kinematics reproduce the dataset's kps2D to <1px. Joint order is
        # identical to self.right/left (out.joints [N,16,3]), so every joint-index map
        # (e.g. _KPS2D_FOR_SMPLX16) is unchanged. CPU-only; not on the training path.
        self.right_full = smplx.create(
            os.path.join(mano_model_folder, "MANO_RIGHT.pkl"), "mano",
            use_pca=False, is_rhand=True, flat_hand_mean=True, num_pca_comps=45,
        )
        self.left_full = smplx.create(
            os.path.join(mano_model_folder, "MANO_LEFT.pkl"), "mano",
            use_pca=False, is_rhand=False, flat_hand_mean=True, num_pca_comps=45,
        )
        if (
            torch.sum(
                torch.abs(
                    self.left_full.shapedirs[:, 0, :] - self.right_full.shapedirs[:, 0, :]
                )
            )
            < 1
        ):
            self.left_full.shapedirs[:, 0, :] *= -1

        self._layer_device = torch.device("cpu")

    def _ensure_device(self, device):
        device = torch.device(device)
        if device != self._layer_device:
            self.left.to(device)
            self.right.to(device)
            self._layer_device = device

    def get_joints_batched(self, params, is_right, device=None):
        """Differentiable, batched MANO joint computation.

        Args:
            params (torch.Tensor): [N, 32] = (t_xyz[3], q_wxyz[4], pose_pca[15], betas[10]).
            is_right (bool): True for right hand, False for left.
            device (torch.device, optional): device to run MANO on (defaults to params.device).

        Returns:
            torch.Tensor: [N, 16, 3] joints with autograd connected to `params`.
        """
        device = torch.device(device) if device is not None else params.device
        self._ensure_device(device)
        params = params.to(device)

        transl        = params[:, 0:3]
        quat_wxyz     = params[:, 3:7]
        hand_pose_pca = params[:, 7:22]
        betas         = params[:, 22:32]

        global_orient = quat_wxyz_to_axis_angle_torch(quat_wxyz)

        layer = self.right if is_right else self.left
        out = layer(
            betas=betas,
            global_orient=global_orient,
            hand_pose=hand_pose_pca,
            transl=transl,
            return_verts=True,
        )
        return out.joints

    def get_vertices_batched(self, params, is_right, device=None):
        """Differentiable, batched MANO vertex computation.

        Args:
            params (torch.Tensor): [N, 32] = (t_xyz[3], q_wxyz[4], pose_pca[15], betas[10]).
            is_right (bool): True for right hand, False for left.
            device (torch.device, optional): device to run MANO on (defaults to params.device).

        Returns:
            torch.Tensor: [N, 778, 3] vertices with autograd connected to `params`.
        """
        device = torch.device(device) if device is not None else params.device
        self._ensure_device(device)
        params = params.to(device)

        transl        = params[:, 0:3]
        quat_wxyz     = params[:, 3:7]
        hand_pose_pca = params[:, 7:22]
        betas         = params[:, 22:32]

        global_orient = quat_wxyz_to_axis_angle_torch(quat_wxyz)

        layer = self.right if is_right else self.left
        out = layer(
            betas=betas,
            global_orient=global_orient,
            hand_pose=hand_pose_pca,
            transl=transl,
            return_verts=True,
        )
        return out.vertices

    def get_joints21_batched(self, params, is_right, device=None):
        """Like get_joints_batched but returns [N,21,3] = 16 MANO joints + 5 fingertips.

        Fingertips are appended in (thumb, index, middle, ring, pinky) order from the
        mesh vertices at the standard MANO tip indices. Used for the 21-joint MPJPE
        that matches the H2O / hand-pose SOTA convention (the tips are the hardest).
        """
        device = torch.device(device) if device is not None else params.device
        self._ensure_device(device)
        params = params.to(device)
        transl, quat_wxyz = params[:, 0:3], params[:, 3:7]
        hand_pose_pca, betas = params[:, 7:22], params[:, 22:32]
        global_orient = quat_wxyz_to_axis_angle_torch(quat_wxyz)
        layer = self.right if is_right else self.left
        out = layer(betas=betas, global_orient=global_orient, hand_pose=hand_pose_pca,
                    transl=transl, return_verts=True)
        tips = out.vertices[:, [745, 317, 444, 556, 673], :]   # thumb,index,middle,ring,pinky
        return torch.cat([out.joints, tips], dim=1)            # [N,21,3]

    def get_joints_full_pose(self, global_aa, pose45, beta, trans, is_right):
        """MANO 3D joints [16,3] (camera frame) from HOI4D's RAW params: 3-dim global
        axis-angle, 45-dim FULL axis-angle hand pose (NO PCA), 10 betas, 3 translation.

        Uses the dedicated use_pca=False / flat_hand_mean=True layer so the forward
        kinematics match HOI4D's manopth exactly (reproduces the dataset kps2D to
        ~0.4px, vs ~12px through the lossy PCA15 path). Joint order is the same MANO
        16-joint kinematic order as get_joints_batched, so _KPS2D_FOR_SMPLX16 and every
        other joint-index map are unchanged. CPU-side (preprocessing); the *_full layers
        are never moved by _ensure_device, so this stays off the training device path.
        """
        layer = self.right_full if is_right else self.left_full
        out = layer(
            betas=torch.as_tensor(beta, dtype=torch.float32).reshape(1, -1),
            global_orient=torch.as_tensor(global_aa, dtype=torch.float32).reshape(1, 3),
            hand_pose=torch.as_tensor(pose45, dtype=torch.float32).reshape(1, 45),
            transl=torch.as_tensor(trans, dtype=torch.float32).reshape(1, 3),
            return_verts=True,
        )
        return out.joints[0].detach().cpu().numpy()           # [16,3] camera frame

    def get_mesh(self, hand_data, is_right):
        """Generate mesh from raw JSONL hand data dict.

        Args:
            hand_data: dict with 'pose', 'wrist_xform' (t_xyz, q_wxyz), 'betas'
            is_right: bool

        Returns:
            vertices: (778, 3) numpy array in world coordinates
            faces: (F, 3) numpy int array
        """
        # CPU-side visualization path; make sure the layer isn't still on the
        # training device from a previous get_joints_batched call.
        self._ensure_device(torch.device("cpu"))

        betas = torch.tensor([hand_data["betas"]], dtype=torch.float32)
        pose = torch.tensor([hand_data["pose"]], dtype=torch.float32)

        wrist = hand_data["wrist_xform"]
        t_xyz = np.array(wrist["t_xyz"])
        q_wxyz = np.array(wrist["q_wxyz"])

        rotvec = _quat_wxyz_to_rotvec(q_wxyz)

        global_orient = torch.from_numpy(rotvec).unsqueeze(0)
        transl = torch.tensor(t_xyz, dtype=torch.float32).unsqueeze(0)

        layer = self.right if is_right else self.left
        output = layer(
            betas=betas,
            global_orient=global_orient,
            hand_pose=pose,
            transl=transl,
            return_verts=True,
        )

        vertices = output.vertices[0].detach().numpy()
        faces = layer.faces.astype(np.int32)
        return vertices, faces

    def get_mesh_from_params(self, params_32, is_right):
        """Generate mesh from the flat 32-dim training parameter vector.

        Args:
            params_32: tensor [32] = [pos(3), rot_qwxyz(4), pose(15), betas(10)]
            is_right: bool

        Returns:
            vertices: (778, 3) numpy array in world coordinates
            faces: (F, 3) numpy int array
        """
        params = params_32.detach().cpu().float()
        hand_data = {
            "wrist_xform": {
                "t_xyz": params[:3].tolist(),
                "q_wxyz": params[3:7].tolist(),
            },
            "pose": params[7:22].tolist(),
            "betas": params[22:32].tolist(),
        }
        return self.get_mesh(hand_data, is_right)

    def get_joints(self, hand_data, is_right, return_tensor=False):
        """Generate 3D joints.
        If return_tensor=True, it stays a Torch tensor (for training loss).
        If False, returns a numpy array (for dataset/vis).
        """
        # This numpy/dict-based path is CPU-only; sync the layer so it can't
        # be mid-flight on CUDA from a prior get_joints_batched call.
        self._ensure_device(torch.device("cpu"))

        betas = torch.as_tensor([hand_data["betas"]], dtype=torch.float32)
        pose = torch.as_tensor([hand_data["pose"]], dtype=torch.float32)

        wrist = hand_data["wrist_xform"]
        t_xyz = np.array(wrist["t_xyz"])
        q_wxyz = np.array(wrist["q_wxyz"])
        rotvec = _quat_wxyz_to_rotvec(q_wxyz)

        global_orient = torch.from_numpy(rotvec).unsqueeze(0)
        transl = torch.tensor(t_xyz, dtype=torch.float32).unsqueeze(0)

        layer = self.right if is_right else self.left

        output = layer(
            betas=betas,
            global_orient=global_orient,
            hand_pose=pose,
            transl=transl,
            return_verts=True,
        )

        joints = output.joints # [21, 3]

        # if output.joints:
        #     joints = output.joints[0] # [21, 3]
        # else:
        #     # Fallback if no joints are found
        #     joints = torch.zeros((1, 21, 3), device=betas.device)
        return joints if return_tensor else joints.detach().cpu().numpy()

    def get_joints_from_tensor(self, params_32, is_right, return_tensor=False):
        """Helper to convert the flat 32-dim vector directly to joints."""
        hand_data = {
            "wrist_xform": {
                "t_xyz": params_32[:3].tolist(),
                "q_wxyz": params_32[3:7].tolist(),
            },
            "pose": params_32[7:22].tolist(),
            "betas": params_32[22:32].tolist(),
        }
        return self.get_joints(hand_data, is_right, return_tensor=return_tensor)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_camera_calibration_from_models_json(json_path, camera_label="camera-rgb"):
    """Build the same (T_device_camera, CameraCalibration) from ground_truth/camera_models.json.

    WHY THIS EXISTS. load_camera_calibration below reads mps_slam_calibration/online_calibration.jsonl,
    which ships only in HOT3D's `mps_artifacts` group (89 GB). The `ground_truth` group we already
    have (4.8 MB/seq) carries the identical quantities for camera-rgb:
        projectionModelType : "CameraModelType.FISHEYE624"
        projectionParams    : the 15 FISHEYE624 params, params[0] = focal
        T_Device_Camera     : quaternion_wxyz + translation_xyz
        imageWidth/Height   : 1408 x 1408
    Cross-check that this is the right source: params[0] is 609.7857 on P0001_10a27bf7, exactly the
    f=609.78 of the validated historical run (preprocess_gb10.sh, pinhole_f609, 50.08 mm MPJPE).

    Is the STATIC calibration acceptable in place of the ONLINE one? Yes, because the online path
    already treats calibration as static: it reads only `f.readline()`, i.e. the first entry, and
    applies it to the whole sequence. So both paths use one fixed calibration per sequence.
    """
    import json as _json

    from projectaria_tools.core.calibration import CameraCalibration, FISHEYE624
    from projectaria_tools.core.sophus import SE3

    with open(json_path) as f:
        cams = _json.load(f)
    for cam in cams:
        if cam.get("label") != camera_label:
            continue
        model = str(cam.get("projectionModelType", ""))
        if "FISHEYE624" not in model:
            raise RuntimeError(
                f"{camera_label} in {json_path} is {model}, not FISHEYE624; this loader would "
                f"silently mis-undistort it.")
        params = np.array(cam["projectionParams"], dtype=np.float64)
        t_dc = np.array(cam["T_Device_Camera"]["translation_xyz"], dtype=np.float64)
        # online_calibration stores UnitQuaternion as [w, [x,y,z]]; this file stores a flat
        # [w,x,y,z], so split it the same way before handing it to SE3.
        q = cam["T_Device_Camera"]["quaternion_wxyz"]
        T_device_camera = SE3.from_quat_and_translation(
            float(q[0]), np.array(q[1:4], dtype=np.float64), t_dc)[0]
        cam_calib = CameraCalibration(
            camera_label, FISHEYE624, params, T_device_camera,
            int(cam.get("imageWidth", 1408)), int(cam.get("imageHeight", 1408)),
            None, 3.14159, "",
        )
        return T_device_camera, cam_calib
    raise RuntimeError(f"Camera '{camera_label}' not found in {json_path}")


def load_camera_calibration(jsonl_path, camera_label="camera-rgb"):
    from projectaria_tools.core.calibration import CameraCalibration, FISHEYE624
    from projectaria_tools.core.sophus import SE3
    """Load camera-rgb calibration from online_calibration.jsonl (first entry)."""
    with open(jsonl_path) as f:
        entry = json.loads(f.readline())
        for cam in entry["CameraCalibrations"]:
            if cam["Label"] == camera_label:
                params = np.array(cam["Projection"]["Params"], dtype=np.float64)
                t_dc = np.array(cam["T_Device_Camera"]["Translation"])
                q_dc = cam["T_Device_Camera"]["UnitQuaternion"]
                T_device_camera = SE3.from_quat_and_translation(
                    q_dc[0], np.array(q_dc[1]), t_dc
                )[0]
                cam_calib = CameraCalibration(
                    camera_label, FISHEYE624, params, T_device_camera,
                    1408, 1408, None, 3.14159, "",
                )
                return T_device_camera, cam_calib
    raise RuntimeError(f"Camera '{camera_label}' not found in calibration")


def load_headset_trajectory(csv_path):
    """Load ground truth headset trajectory (timecode domain).
    Returns dict: timecode_ns -> (translation, quaternion_wxyz).
    """
    poses = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts_ns = int(row["timestamp[ns]"])
            t = np.array([
                float(row["t_wo_x[m]"]),
                float(row["t_wo_y[m]"]),
                float(row["t_wo_z[m]"]),
            ])
            q_wxyz = np.array([
                float(row["q_wo_w"]),
                float(row["q_wo_x"]),
                float(row["q_wo_y"]),
                float(row["q_wo_z"]),
            ])
            poses[ts_ns] = (t, q_wxyz)
    return poses


def load_hand_poses(jsonl_path):
    """Load MANO hand poses from JSONL. Returns dict: timecode_ns -> hand_data."""
    hand_poses = {}
    with open(jsonl_path) as f:
        for line in f:
            entry = json.loads(line)
            hand_poses[entry["timestamp_ns"]] = entry["hand_poses"]
    return hand_poses


def find_closest(sorted_keys, query):
    """Find the closest key in a sorted list."""
    idx = bisect.bisect_left(sorted_keys, query)
    if idx == 0:
        return sorted_keys[0]
    if idx >= len(sorted_keys):
        return sorted_keys[-1]
    before = sorted_keys[idx - 1]
    after = sorted_keys[idx]
    return before if (query - before) <= (after - query) else after


# ---------------------------------------------------------------------------
# Projection and rendering
# ---------------------------------------------------------------------------

def project_vertices(vertices_world, T_world_device, T_device_camera, cam_calib,
                     image_width=1408):
    """Project 3D world vertices to 2D pixel coordinates.

    Applies 90 deg CW rotation to match MP4 video orientation.
    """
    T_camera_world = T_device_camera.inverse().to_matrix() @ T_world_device.inverse().to_matrix()

    N = vertices_world.shape[0]
    verts_homo = np.hstack([vertices_world, np.ones((N, 1))])
    verts_cam = (T_camera_world @ verts_homo.T).T[:, :3]

    depths = verts_cam[:, 2]
    pixels = np.zeros((N, 2))
    valid = np.zeros(N, dtype=bool)
    margin = 100

    for i in np.where(depths > 0.01)[0]:
        p = cam_calib.project(verts_cam[i])
        if p is not None:
            u = (image_width - 1) - p[1]
            v = p[0]
            if -margin <= u <= image_width + margin and -margin <= v <= image_width + margin:
                pixels[i] = [u, v]
                valid[i] = True

    return pixels, depths, valid


def project_joints_torch(joints_world, T_camera_world, focal_length, cx, cy, image_width=1408):
    """Differentiable pinhole approximation of project_vertices.

    Applies the same 90° CW coordinate flip as project_vertices so that projected
    pixel coordinates are directly comparable to GT pixels produced by that function.
    Fisheye distortion is ignored — the approximation is sufficient for gradient
    direction during training.

    Args:
        joints_world (torch.Tensor): Shape [B, ..., 3] — 3D joint positions in world
            coordinates.
        T_camera_world (torch.Tensor): Shape [B, 4, 4] — world-to-camera extrinsic
            matrix for each item in the leading batch dimension.
        focal_length (torch.Tensor | float): Shape [B] or scalar — focal length f
            (assumes fx == fy for the spherical fisheye approximation).
        cx (torch.Tensor | float): Shape [B] or scalar — principal point x.
        cy (torch.Tensor | float): Shape [B] or scalar — principal point y.
        image_width (int): Sensor width in pixels used by project_vertices (default 1408).

    Returns:
        torch.Tensor: Shape [B, ..., 2] — projected [u, v] pixel coordinates.
    """
    B = joints_world.shape[0]
    device = joints_world.device
    dtype = joints_world.dtype
    T = T_camera_world.to(device=device, dtype=dtype)       # [B, 4, 4]

    ones = torch.ones(*joints_world.shape[:-1], 1, device=device, dtype=dtype)
    j_homo = torch.cat([joints_world, ones], dim=-1)        # [B, ..., 4]

    # Flatten middle dims into M so we can use batched matmul
    leading = j_homo.shape[1:-1]                            # (...) dims tuple
    j_flat = j_homo.reshape(B, -1, 4)                      # [B, M, 4]
    # T @ each column vector: T [B,4,4] × j_flat.T [B,4,M] → [B,4,M] → [B,M,4]
    j_cam = (T @ j_flat.transpose(1, 2)).transpose(1, 2)   # [B, M, 4]
    j_cam = j_cam[..., :3].reshape(B, *leading, 3)         # [B, ..., 3]

    z = j_cam[..., 2].clamp(min=1e-4)

    # Broadcast per-batch intrinsics over all leading dims
    n_leading = len(leading)
    if isinstance(focal_length, torch.Tensor) and focal_length.ndim > 0:
        f   = focal_length.to(device=device, dtype=dtype).view(B, *([1] * n_leading))
        cx_ = cx.to(device=device, dtype=dtype).view(B, *([1] * n_leading))
        cy_ = cy.to(device=device, dtype=dtype).view(B, *([1] * n_leading))
    else:
        f, cx_, cy_ = focal_length, cx, cy

    col = f * j_cam[..., 0] / z + cx_
    row = f * j_cam[..., 1] / z + cy_

    # Match project_vertices: 90° CW rotation (col, row) → (W-1-row, col)
    u = (image_width - 1) - row
    v = col

    return torch.stack([u, v], dim=-1)


def project_vertices_camera_space(vertices_camera, cam_calib, image_width=1408):
    """Project 3D camera-space vertices to 2D pixel coordinates.

    Same as project_vertices but skips the world→camera transform because the
    vertices are already in camera space (e.g. from HandCropHead's
    _crop_relative_to_global output).
    """
    N = vertices_camera.shape[0]
    depths = vertices_camera[:, 2]
    pixels = np.zeros((N, 2))
    valid = np.zeros(N, dtype=bool)
    margin = 100

    for i in np.where(depths > 0.01)[0]:
        p = cam_calib.project(vertices_camera[i])
        if p is not None:
            u = (image_width - 1) - p[1]
            v = p[0]
            if -margin <= u <= image_width + margin and -margin <= v <= image_width + margin:
                pixels[i] = [u, v]
                valid[i] = True

    return pixels, depths, valid


def render_mesh_overlay(image, pixels, faces, depths, valid, color, alpha, wireframe):
    """Render a mesh overlay on the image using filled triangles with alpha blending."""
    overlay = image.copy()

    face_depths = []
    valid_faces = []
    for i, face in enumerate(faces):
        if valid[face[0]] and valid[face[1]] and valid[face[2]]:
            face_depths.append(
                (depths[face[0]] + depths[face[1]] + depths[face[2]]) / 3.0
            )
            valid_faces.append(i)

    if not valid_faces:
        return image

    sorted_indices = np.argsort([-d for d in face_depths])

    for idx in sorted_indices:
        face = faces[valid_faces[idx]]
        pts = pixels[face].astype(np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(overlay, [pts], color)

    result = cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0)

    if wireframe:
        edge_color = tuple(max(0, int(c * 0.6)) for c in color)
        for idx in sorted_indices:
            face = faces[valid_faces[idx]]
            pts = pixels[face].astype(np.int32)
            for j in range(3):
                cv2.line(
                    result, tuple(pts[j]), tuple(pts[(j + 1) % 3]),
                    edge_color, 1, cv2.LINE_AA,
                )

    return result


# ---------------------------------------------------------------------------
# High-level API for training integration
# ---------------------------------------------------------------------------

GT_LEFT_COLOR = (255, 150, 50)    # blue-ish (BGR)
GT_RIGHT_COLOR = (50, 50, 255)    # red-ish (BGR)
PRED_LEFT_COLOR = (50, 255, 50)   # green (BGR)
PRED_RIGHT_COLOR = (50, 165, 255) # orange (BGR)


def setup_vis_context(seq_path, mano_model_folder=None, mano_model=None):
    """One-time setup: load camera calibration, headset trajectory, hand poses.

    Uses the same data as the standalone visualize_mano_hands.py script.
    Returns a context dict, or None if required files are missing.
    """
    video_path = os.path.join(seq_path, "video_main_rgb.mp4")
    jsonl_path = os.path.join(seq_path, "hand_data", "mano_hand_pose_trajectory.jsonl")
    calib_path = os.path.join(seq_path, "mps_slam_calibration", "online_calibration.jsonl")
    headset_path = os.path.join(seq_path, "ground_truth", "headset_trajectory.csv")

    for p in [video_path, jsonl_path, calib_path, headset_path]:
        if not os.path.exists(p):
            print(f"[VIS] Skipping visualization: missing {p}")
            return None

    if mano_model is None:
        mano_model = MANOModel(mano_model_folder)
    T_device_camera, cam_calib = load_camera_calibration(calib_path)
    headset_poses = load_headset_trajectory(headset_path)
    headset_ts_sorted = sorted(headset_poses.keys())

    hand_poses = load_hand_poses(jsonl_path)
    hand_ts_sorted = sorted(hand_poses.keys())

    n_video = len(VideoReader(video_path))

    return {
        "mano_model": mano_model,
        "T_device_camera": T_device_camera,
        "cam_calib": cam_calib,
        "headset_poses": headset_poses,
        "headset_ts_sorted": headset_ts_sorted,
        "hand_poses": hand_poses,
        "hand_ts_sorted": hand_ts_sorted,
        "n_video": n_video,
        "video_path": video_path,
    }


def _frame_to_timecode(frame_idx, n_video, hand_ts_sorted):
    """Map video frame index to timecode via linear interpolation.

    Same approach as the standalone visualize_mano_hands.py script.
    """
    ts_start = hand_ts_sorted[0]
    ts_end = hand_ts_sorted[-1]
    frac = frame_idx / max(n_video - 1, 1)
    return int(ts_start + frac * (ts_end - ts_start))


def render_hand_comparison(vis_context, frame_idx, gt_params, pred_params):
    from projectaria_tools.core.sophus import SE3
    """Render GT and predicted MANO hands overlaid on a full-resolution frame.

    GT hands are rendered from the raw JSONL data (matching the standalone script's
    approach for correct alignment). Predicted hands use the model output vector.

    Args:
        vis_context: dict from setup_vis_context
        frame_idx: video frame index
        gt_params: tensor [64] ground truth (unused for GT mesh, used only as fallback)
        pred_params: tensor [64] predicted (2 hands x 32)

    Returns:
        RGB numpy image (H, W, 3) uint8 for wandb.Image, or None on failure.
    """
    try:
        cap = cv2.VideoCapture(vis_context["video_path"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, image = cap.read()
        cap.release()
        if not ret:
            print(f"[VIS] Failed to read frame {frame_idx}")
            return None
    except Exception as e:
        print(f"[VIS] Failed to read frame {frame_idx}: {e}")
        return None

    # Map frame to timecode using linear interpolation (same as standalone script)
    query_tc = _frame_to_timecode(
        frame_idx, vis_context["n_video"], vis_context["hand_ts_sorted"]
    )

    # Find closest headset pose
    closest_headset_ts = find_closest(vis_context["headset_ts_sorted"], query_tc)
    t_wd, q_wd_wxyz = vis_context["headset_poses"][closest_headset_ts]
    T_world_device = SE3.from_quat_and_translation(
        q_wd_wxyz[0], q_wd_wxyz[1:], t_wd
    )[0]

    # Find closest hand pose from raw JSONL data
    closest_hand_ts = find_closest(vis_context["hand_ts_sorted"], query_tc)
    hand_data = vis_context["hand_poses"][closest_hand_ts]

    mano = vis_context["mano_model"]
    T_dev_cam = vis_context["T_device_camera"]
    cam_calib = vis_context["cam_calib"]

    # Render GT hands from raw JSONL data (solid fill)
    for is_right, color in [(False, GT_LEFT_COLOR), (True, GT_RIGHT_COLOR)]:
        hand_key = str(1 if is_right else 0)
        if hand_key not in hand_data:
            continue
        try:
            verts, faces = mano.get_mesh(hand_data[hand_key], is_right)
            pixels, depths, valid = project_vertices(verts, T_world_device, T_dev_cam, cam_calib)
            if valid.sum() >= 10:
                image = render_mesh_overlay(image, pixels, faces, depths, valid, color, 0.35, False)
        except Exception as e:
            print(f"[VIS] GT {'right' if is_right else 'left'} failed: {e}")

    # Render predicted hands from model output (wireframe).
    # Predicted params are in camera space (from HandCropHead's
    # _crop_relative_to_global), so project directly without the
    # world→camera transform.
    for is_right, color in [(False, PRED_LEFT_COLOR), (True, PRED_RIGHT_COLOR)]:
        offset = 32 if is_right else 0
        params = pred_params[offset:offset + 32]
        if params.abs().sum() < 1e-6:
            continue
        try:
            verts, faces = mano.get_mesh_from_params(params, is_right)
            pixels, depths, valid = project_vertices_camera_space(verts, cam_calib)
            if valid.sum() >= 10:
                image = render_mesh_overlay(image, pixels, faces, depths, valid, color, 0.35, True)
        except Exception as e:
            print(f"[VIS] Pred {'right' if is_right else 'left'} failed: {e}")

    # Add legend
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, "GT Left", (10, 30), font, 0.7, GT_LEFT_COLOR, 2)
    cv2.putText(image, "GT Right", (10, 60), font, 0.7, GT_RIGHT_COLOR, 2)
    cv2.putText(image, "Pred Left", (10, 90), font, 0.7, PRED_LEFT_COLOR, 2)
    cv2.putText(image, "Pred Right", (10, 120), font, 0.7, PRED_RIGHT_COLOR, 2)

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def hand_joints_to_rgb_points(gt_hj, pred_hj, gt_rgb=(0, 200, 0), pred_rgb=(220, 30, 30)):
    """Build a coloured 3D point set comparing GT vs predicted hand joints.

    The 2D overlays hide the depth axis -- which is exactly where the placement
    error lives -- so this feeds a 3D scatter (e.g. wandb.Object3D) where GT
    joints are green and predictions red, making the depth/placement residual
    directly visible.

    Args:
        gt_hj, pred_hj: array-likes of shape [H, J, 3] (or any [..., 3])
            camera-frame joints in metres, same shape/ordering for both.
        gt_rgb, pred_rgb: 0-255 RGB triples.

    Returns:
        (M, 6) float32 array of [x, y, z, r, g, b], GT points then pred points,
        or None when shapes mismatch or no valid joints remain. Absent-hand
        fillers (all-zero GT) and non-finite joints are dropped; predictions are
        kept only where the corresponding GT is valid, so green/red points stay
        one-to-one.
    """
    gt = np.asarray(gt_hj, dtype=np.float32).reshape(-1, 3)
    pred = np.asarray(pred_hj, dtype=np.float32).reshape(-1, 3)
    if gt.shape != pred.shape or gt.size == 0:
        return None
    valid = (
        np.isfinite(gt).all(axis=1)
        & np.isfinite(pred).all(axis=1)
        & (np.linalg.norm(gt, axis=1) > 1e-6)
    )
    if not valid.any():
        return None
    gt, pred = gt[valid], pred[valid]
    gt_pts = np.concatenate([gt, np.tile(gt_rgb, (len(gt), 1))], axis=1)
    pred_pts = np.concatenate([pred, np.tile(pred_rgb, (len(pred), 1))], axis=1)
    return np.concatenate([gt_pts, pred_pts], axis=0).astype(np.float32)
