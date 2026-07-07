import cv2
import matplotlib.pyplot as plt
import numpy as np

from perception.visualization.geometry import project_3d_box_to_image

_CAR_OBJ         = "multi_object_tracking/viewer/car.obj"
_OBJ_NATIVE      = np.array([8.95, 3.71, 2.97], dtype=np.float32)
_VEHICLE_CLASSES = {"Car", "Van", "Truck"}

_BOX_EDGES = [
    (0,1),(1,2),(2,3),(3,0),
    (4,5),(5,6),(6,7),(7,4),
    (0,4),(1,5),(2,6),(3,7),
]


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
    threshold=4,
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
    dataset        : KittiDetectionDataset
    frames         : iterable of int
    final_det_ids  : list[ndarray]   per-frame confirmed track IDs
    threshold      : float           minimum score for unconfirmed detections
    out_file       : str             output path
    fps            : int
    output_height  : int             pixel height of each panel
    """
    frames = list(frames)
    cmap   = plt.get_cmap("tab20")
    H      = output_height

    obj_verts, obj_edges = _load_obj_crease_edges(_CAR_OBJ, crease_angle_deg=50)

    first_img      = dataset[frames[0]]["image"]
    src_h, src_w   = first_img.shape[:2]
    cam_w          = int(src_w * H / src_h)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_file, fourcc, fps, (cam_w, 2 * H))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for '{out_file}'")

    print(f"Rendering {len(frames)} frames to '{out_file}' …")
    for idx, i in enumerate(frames):
        data = dataset[i]

        P2            = data["P2"]
        V2C           = data["V2C"]
        points        = data["points"]
        image         = data["image"]
        objects_cam   = data["objects_cam"]
        objects_lidar = data["objects"]
        det_scores    = data["scores"]

        confirmed_mask = final_det_ids[i] > 0
        score_mask     = det_scores > threshold
        vis_mask       = confirmed_mask | score_mask

        objects_cam_v = objects_cam[vis_mask]
        objects_lid_v = np.array(objects_lidar)[vis_mask]
        det_ids       = final_det_ids[i][vis_mask]
        names_v       = np.array(data["names"])[vis_mask]

        def _draw_boxes(img):
            for obj_cam, track_id in zip(objects_cam_v, det_ids):
                if track_id == 0:
                    continue
                color = (np.array(cmap(track_id % cmap.N)) * 255).astype(np.uint8)
                bgr   = (int(color[2]), int(color[1]), int(color[0]))
                corners_2d = project_3d_box_to_image(obj_cam, P2, image.shape)
                if corners_2d is None:
                    continue
                for s, e in _BOX_EDGES:
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

        for obj_cam, obj_lid, track_id, name in zip(
            objects_cam_v, objects_lid_v, det_ids, names_v
        ):
            if track_id == 0:
                continue
            color = (np.array(cmap(track_id % cmap.N)) * 255).astype(np.uint8)
            bgr   = (int(color[2]), int(color[1]), int(color[0]))
            h_b, w_b, l_b, x_b, y_b, z_b, ry = obj_lid.tolist()
            if l_b * w_b * h_b <= 0:
                continue

            if name in _VEHICLE_CLASSES:
                # Project scaled OBJ crease-edge wireframe into the LiDAR depth panel
                yaw      = -ry - np.pi / 2
                c_y, s_y = np.cos(yaw), np.sin(yaw)
                v3d      = obj_verts * (np.array([l_b, w_b, h_b]) / _OBJ_NATIVE)
                R_z      = np.array([[c_y, -s_y, 0.0], [s_y, c_y, 0.0], [0.0, 0.0, 1.0]],
                                    dtype=np.float32)
                v3d  = v3d @ R_z.T
                v3d += np.array([x_b, y_b, z_b + h_b / 2], dtype=np.float32)
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
                corners_2d = project_3d_box_to_image(obj_cam, P2, lidar_img.shape)
                if corners_2d is not None:
                    for s, e in _BOX_EDGES:
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
