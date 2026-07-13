import cv2
import matplotlib.pyplot as plt
import numpy as np

from perception.boxes import BOX_EDGES
from perception.visualization.common import (
    CAR_OBJ_NATIVE_SIZE,
    CAR_OBJ_PATH,
    MESH_CLASSES,
    select_visible,
    track_color,
)
from perception.visualization.geometry import project_box_to_image


def _load_obj_crease_edges(path, crease_angle_deg=50):
    """Parse an OBJ file and return vertices + hard (crease) edges.

    Only edges where adjacent faces meet at a dihedral angle greater than
    crease_angle_deg are kept, giving a clean wireframe.

    Returns
    -------
    verts : ndarray (V, 3)
    edges : list of (int, int)
    """
    verts, raw_faces, tris = [], [], []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == 'v':
                verts.append([float(p) for p in parts[1:4]])
            elif parts[0] == 'f':
                idx = [int(p.split('/')[0]) - 1 for p in parts[1:]]
                raw_faces.append(idx)
                for k in range(1, len(idx) - 1):
                    tris.append((idx[0], idx[k], idx[k + 1]))

    verts = np.array(verts, dtype=np.float32)
    tris  = np.array(tris,  dtype=np.int32)

    v0, v1, v2 = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    normals     = np.cross(v1 - v0, v2 - v0).astype(np.float32)
    normals    /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8

    edge_to_tris: dict = {}
    for ti, tri in enumerate(tris):
        for k in range(3):
            e = tuple(sorted((int(tri[k]), int(tri[(k + 1) % 3]))))
            edge_to_tris.setdefault(e, []).append(ti)

    raw_edges: set = set()
    for face in raw_faces:
        n = len(face)
        for k in range(n):
            raw_edges.add(tuple(sorted((face[k], face[(k + 1) % n]))))

    cos_thresh = np.cos(np.deg2rad(crease_angle_deg))
    hard_edges = []
    for e in raw_edges:
        adj = edge_to_tris.get(e, [])
        if len(adj) <= 1:
            hard_edges.append(e)
        elif np.dot(normals[adj[0]], normals[adj[1]]) < cos_thresh:
            hard_edges.append(e)

    return verts, hard_edges


def create_tracking_video(
    dataset,
    frames,
    final_det_ids,
    show_unconfirmed_above=4.0,
    out_file="tracking.mp4",
    fps=10,
    output_height=480,
):
    """Render tracked detections to an MP4 with camera + LiDAR depth panels.

    Top panel:    camera image with coloured 3D bounding box overlays.
    Bottom panel: LiDAR points projected through the same camera matrix,
                  coloured by depth, with car mesh wireframes.

    Parameters
    ----------
    dataset        : SequenceDataset
    frames         : iterable of int
    final_det_ids  : list[ndarray]   per-frame confirmed track IDs
    show_unconfirmed_above : float
        Also draw unconfirmed detections scoring above this (default 4 —
        raw-logit scale, so sigmoid-scored detections are never drawn).
    out_file       : str             output path
    fps            : int
    output_height  : int             pixel height of each panel
    """
    frames = list(frames)
    H      = output_height

    obj_verts, obj_edges = _load_obj_crease_edges(CAR_OBJ_PATH, crease_angle_deg=50)

    first_img      = dataset[frames[0]].image
    src_h, src_w   = first_img.shape[:2]
    cam_w          = int(src_w * H / src_h)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_file, fourcc, fps, (cam_w, 2 * H))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for '{out_file}'")

    print(f"Rendering {len(frames)} frames to '{out_file}' …")
    for idx, i in enumerate(frames):
        frame = dataset[i]

        P2     = frame.camera.projection
        V2C    = frame.camera.lidar_to_cam
        points = frame.points
        image  = frame.image

        boxes_v, det_ids, names_v = select_visible(
            frame.detections, final_det_ids[i], show_unconfirmed_above
        )

        def _draw_boxes(img):
            for box, track_id in zip(boxes_v, det_ids):
                if track_id == 0:
                    continue
                color = track_color(track_id)
                bgr   = (int(color[2]), int(color[1]), int(color[0]))
                corners_2d = project_box_to_image(box, V2C, P2, image.shape)
                if corners_2d is None:
                    continue
                for s, e in BOX_EDGES:
                    cv2.line(img,
                             tuple(corners_2d[s].astype(int)),
                             tuple(corners_2d[e].astype(int)),
                             bgr, 1, cv2.LINE_AA)
                cx = int(corners_2d[:, 0].mean())
                cy = max(int(corners_2d[:, 1].min()) - 4, 10)
                cv2.putText(img, str(track_id), (cx, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, bgr, 1, cv2.LINE_AA)

        # ── Camera panel ──────────────────────────────────────────────────────
        cam_img = image.copy()
        _draw_boxes(cam_img)
        cam_panel = cv2.resize(cam_img, (cam_w, H))

        # ── LiDAR depth panel ─────────────────────────────────────────────────
        lidar_img = np.zeros((src_h, src_w, 3), dtype=np.uint8)
        pts_hom   = np.hstack([points[:, :3], np.ones((len(points), 1))])
        pts_cam   = (V2C @ pts_hom.T).T
        depth     = pts_cam[:, 2]
        in_front  = depth > 0
        pts_cam, depth = pts_cam[in_front], depth[in_front]
        pts_img   = (P2 @ pts_cam.T).T
        px = (pts_img[:, 0] / pts_img[:, 2]).astype(np.float32)
        py = (pts_img[:, 1] / pts_img[:, 2]).astype(np.float32)
        in_bounds = (px >= 0) & (px < src_w) & (py >= 0) & (py < src_h)
        px, py, depth = px[in_bounds].astype(int), py[in_bounds].astype(int), depth[in_bounds]
        order   = np.argsort(-depth)
        px, py, depth = px[order], py[order], depth[order]
        depth_n = np.clip(1.0 - depth / 40.0, 0.0, 1.0)
        rgba    = (plt.cm.plasma(depth_n)[:, :3] * 255).astype(np.uint8)
        lidar_img[py, px, 0] = rgba[:, 2]
        lidar_img[py, px, 1] = rgba[:, 1]
        lidar_img[py, px, 2] = rgba[:, 0]

        for box, track_id, name in zip(boxes_v, det_ids, names_v):
            if track_id == 0:
                continue
            color = track_color(track_id)
            bgr   = (int(color[2]), int(color[1]), int(color[0]))
            x_b, y_b, z_b, l_b, w_b, h_b, yaw = box.tolist()
            if l_b * w_b * h_b <= 0:
                continue

            if name in MESH_CLASSES:
                # Project scaled OBJ crease-edge wireframe into the LiDAR depth panel
                c_y, s_y = np.cos(yaw), np.sin(yaw)
                v3d      = obj_verts * (np.array([l_b, w_b, h_b]) / CAR_OBJ_NATIVE_SIZE)
                R_z      = np.array([[c_y, -s_y, 0.0], [s_y, c_y, 0.0], [0.0, 0.0, 1.0]],
                                    dtype=np.float32)
                v3d  = v3d @ R_z.T
                v3d += np.array([x_b, y_b, z_b], dtype=np.float32)
                v_hom = np.hstack([v3d, np.ones((len(v3d), 1))])
                v_cam = (V2C @ v_hom.T).T
                v_img = (P2 @ v_cam.T).T
                in_fv = v_cam[:, 2] > 0
                vpx   = np.where(in_fv, v_img[:, 0] / v_img[:, 2], -1.0).astype(np.float32)
                vpy   = np.where(in_fv, v_img[:, 1] / v_img[:, 2], -1.0).astype(np.float32)
                for vi, vj in obj_edges:
                    if not (in_fv[vi] and in_fv[vj]):
                        continue
                    cv2.line(lidar_img,
                             (int(vpx[vi]), int(vpy[vi])),
                             (int(vpx[vj]), int(vpy[vj])),
                             bgr, 1, cv2.LINE_AA)
            else:
                # Cyclist / Pedestrian — project the 3D box corners (same as camera panel)
                corners_2d = project_box_to_image(box, V2C, P2, lidar_img.shape)
                if corners_2d is not None:
                    for s, e in BOX_EDGES:
                        cv2.line(lidar_img,
                                 tuple(corners_2d[s].astype(int)),
                                 tuple(corners_2d[e].astype(int)),
                                 bgr, 1, cv2.LINE_AA)

        lidar_panel = cv2.resize(lidar_img, (cam_w, H))
        writer.write(np.vstack([cam_panel, lidar_panel]))

        if (idx + 1) % 50 == 0:
            print(f"  {idx + 1}/{len(frames)} frames rendered")

    writer.release()
    print(f"Video saved → '{out_file}'")
