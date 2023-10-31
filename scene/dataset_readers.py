#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import io
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readE57SceneInfo(path, eval, llffhold=8):
    import pye57
    import quaternion
    from tqdm import tqdm
    import open3d as o3d
    from collections import deque

    e57 = pye57.E57(path)

    path = Path(path)
    extract_path = path.parent / f"{path.stem}_scene_info"
    os.makedirs(extract_path, exist_ok=True)
    os.makedirs(extract_path / "images", exist_ok=True)

    # 1. Extract all cameras
    cam_infos = []

    for idx, image in tqdm(enumerate(e57.root["images2D"]), desc="Processing images", total = e57.scan_count * 6):
        if (idx + 1) % 6 == 0 or (idx + 2) % 6 == 0: continue
        # image_path = os.path.join(path, cam_name)
        # image_name = Path(cam_name).stem
        jpeg_buffer = image["pinholeRepresentation"]["jpegImage"].read_buffer()
        image_data = Image.open(io.BytesIO(jpeg_buffer))
        if not (extract_path / f"images/{idx}.jpg").exists():
            with open(extract_path / f"images/{idx}.jpg", "wb") as fh:
                fh.write(jpeg_buffer)

        # Camera intrinsic matrix
        focal_length = image["pinholeRepresentation"]["focalLength"].value() # Note: In meters, must be translated to pixels
        image_width = image["pinholeRepresentation"]["imageWidth"].value()
        image_height = image["pinholeRepresentation"]["imageHeight"].value()
        pixel_width = image["pinholeRepresentation"]["pixelWidth"].value()
        pixel_height = image["pinholeRepresentation"]["pixelHeight"].value()

        FovX = focal2fov(focal_length / pixel_width, image_width)
        FovY = focal2fov(focal_length / pixel_height, image_height)

        R = quaternion.as_rotation_matrix(np.quaternion(
            image["pose"]["rotation"]["w"].value(),
            image["pose"]["rotation"]["x"].value(),
            image["pose"]["rotation"]["y"].value(),
            image["pose"]["rotation"]["z"].value(),
        ))

        T = np.array([
            image["pose"]["translation"]["x"].value(),
            image["pose"]["translation"]["y"].value(),
            image["pose"]["translation"]["z"].value(),
        ])

        extrinsic = np.column_stack((R, T))
        extrinsic = np.row_stack((extrinsic, [0, 0, 0, 1]))

        # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        extrinsic[:3, 1:3] *= -1

        # Specify camera's pose directly, instead of how world is transformed around camera.
        extrinsic = np.linalg.inv(extrinsic)

        # R is stored transposed due to 'glm' in CUDA code
        R = np.transpose(extrinsic[:3,:3])
        T = extrinsic[:3, 3]

        cam_infos.append(CameraInfo(
            uid=idx,
            R=R, T=T, FovX=FovX, FovY=FovY,
            image=image_data, image_path=f"{extract_path}/images/{idx}.jpg", image_name=f"{idx}.jpg",
            width=image_width, height=image_height))

    # 2. Split cameras into train/test (?)
    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    # 3. Just run getNerfppNorm?
    nerf_normalization = getNerfppNorm(train_cam_infos)

    # 4. Get pointcloud info in BasicPointCloud format
    ply_path = extract_path / "pointcloud.ply"
    pcd = o3d.geometry.PointCloud()
    if not os.path.exists(ply_path):
        points = deque()
        colors = deque()
        for idx in tqdm(range(e57.scan_count), desc="Assembling pointcloud"):
            p, c = point_data(e57, idx)
            points.append(p)
            colors.append(c)

        points = np.concatenate(points)
        colors = np.concatenate(colors)

        print("Simplifying pointcloud")
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        pcd = pcd.voxel_down_sample(voxel_size=0.1)

        print("Caching pointcloud data")
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        pcd = BasicPointCloud(points, colors, normals=np.zeros((len(points), 3)))
        storePly(ply_path, points, colors)

    # pcd = o3d.io.read_point_cloud(str(ply_path))
    # render_ply(pcd, train_cam_infos)
    # sys.exit(0)

    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=str(ply_path)
    )

    print("HELLO E57 <3")

    return scene_info

def render_ply(pcd, cameras):
    """
    pcd = o3d.io.read_point_cloud(str(ply_path))
    render_ply(pcd, train_cam_infos)
    """
    from tqdm import tqdm
    import open3d as o3d

    pcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors) * 255)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    for cam_info in tqdm(cameras, desc="Adding cameras"):
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            cam_info.width, cam_info.height,
            fov2focal(cam_info.FovX, cam_info.width),
            fov2focal(cam_info.FovY, cam_info.height),
            cam_info.width / 2, cam_info.height / 2
        );

        extrinsic = np.column_stack((cam_info.R, cam_info.T))
        extrinsic = np.row_stack((extrinsic, [0, 0, 0, 1]))
        extrinsic[:3,:3] = np.transpose(extrinsic[:3,:3])  # R is stored transposed due to 'glm' in CUDA code

        camera = o3d.geometry.LineSet.create_camera_visualization(intrinsic, extrinsic);
        vis.add_geometry(camera)

    vis.run()
    vis.destroy_window()

def point_data(e57, index):
    header = e57.get_header(index)
    data = e57.read_scan(index, colors=True)

    # Create pointcloud - Should be global
    points = np.column_stack((data["cartesianX"], data["cartesianY"], data["cartesianZ"]))

    try:
        assert header["colorLimits"]["colorRedMaximum"].value() == 255
        assert header["colorLimits"]["colorGreenMaximum"].value() == 255
        assert header["colorLimits"]["colorBlueMaximum"].value() == 255.0
    except:
        pass

    colors = np.column_stack((
        np.array(data["colorRed"]),
        np.array(data["colorGreen"]),
        np.array(data["colorBlue"]),
    ))

    return points, colors

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "E57": readE57SceneInfo,
}
