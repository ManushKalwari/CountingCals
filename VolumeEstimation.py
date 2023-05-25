import os
import cv2
import numpy as np
import open3d as o3d
import trimesh
from PIL import Image



def extract_features(image):
    gray = image.convert("L")
    #gray = cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors




def match_features(descriptors1, descriptors2):
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(descriptors1, descriptors2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.9 * n.distance:
            good_matches.append(m)
    return good_matches





def get_3D_points(keypoints1, keypoints2, matches, camera_matrix):
    src_points = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_points = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)


    essential_matrix, _ = cv2.findEssentialMat(src_points, dst_points, camera_matrix)
    _, rot_matrix, trans_matrix, _ = cv2.recoverPose(essential_matrix, src_points, dst_points, camera_matrix)

    proj_matrix1 = np.hstack((camera_matrix, np.zeros((3, 1))))
    proj_matrix2 = np.hstack((rot_matrix, trans_matrix))

    points_4d = cv2.triangulatePoints(
        proj_matrix1, proj_matrix2, src_points.reshape(-1, 2).T, dst_points.reshape(-1, 2).T)
    points_3d_homogeneous = cv2.convertPointsFromHomogeneous(points_4d.T)
    points_3d = points_3d_homogeneous[:, 0, :]
    
    return points_3d





def integrate_point_clouds(point_clouds):
    #print("integrating point clouds...")
    merged_point_cloud = o3d.geometry.PointCloud()
    
    for point_cloud in point_clouds:
        merged_point_cloud += point_cloud
    return merged_point_cloud




def ballPivot_reconstruction(point_cloud):

    point_cloud.estimate_normals()
    distances = point_cloud.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 2 * avg_dist   

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(point_cloud,
               o3d.utility.DoubleVector([radius, radius * 5]))
    
    # create the triangular mesh with the vertices and faces from open3d
    tri_mesh = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles),
                              vertex_normals=np.asarray(mesh.vertex_normals))
    
    #trimesh.convex.is_convex(tri_mesh)
    #trimesh.exchange.obj.export_obj(tri_mesh) 
    #tri_mesh.show()

    return tri_mesh




def estimate_volume(tri_mesh):

    volume = tri_mesh.volume
    if volume is not None:
        #print(f"Estimated volume: {volume*1000} cubic units")
        return volume*1000
    else:
        #print("Error: Failed to estimate volume.")
        return None
        
    


def getVolume(images_list):
    images = []
    keypoints_list = []
    descriptors_list = []

    for image in images_list:
        #img = cv2.resize(image, (0, 0), fx = 0.5, fy = 0.5)
        #for img_file in img_files:
        img = Image.open(image).resize((500, 500))
        images.append(img)

        if len(images) > 1:
            keypoints, descriptors = extract_features(img)
            keypoints_list.append(keypoints)
            descriptors_list.append(descriptors)

    if len(images) < 3:
        print("At least three images are required for volume estimation.")
        return None

    # Find chessboard corners in the first two images for camera calibration
    gray1 = cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(images[1], cv2.COLOR_BGR2GRAY)

    ret1, corners1 = cv2.findChessboardCorners(gray1, (7, 6), None)
    ret2, corners2 = cv2.findChessboardCorners(gray2, (7, 6), None)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((6 * 7, 3), np.float32)
    objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

    objpoints = [objp, objp]
    corners21 = cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
    corners22 = cv2.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)

    imgpoints = [corners21, corners22]

    ret, camera_matrix, _, _, _ = cv2.calibrateCamera(objpoints, imgpoints, gray1.shape[::-1], None, None)

    point_clouds = []

    for i in range(len(images) - 2):
        matches = match_features(descriptors_list[i], descriptors_list[i + 1])
        points_3d = get_3D_points(keypoints_list[i], keypoints_list[i + 1], matches, camera_matrix)

        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points_3d)
        point_clouds.append(point_cloud)


    integrated_point_cloud = integrate_point_clouds(point_clouds)
    integrated_point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2, max_nn=50))
    mesh = ballPivot_reconstruction(integrated_point_cloud)
    
    volume = estimate_volume(mesh)
    return volume

