import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

# PARAMETER
# front=[ -0.8267451277297444, -0.25936864732022502, 0.49921978983450321 ]
# lookat=[ 0.72116490107331344, 2.8112010782972727, -4.9213092248048866 ]
# up=[ 0.46251535441024405, 0.19181020308112595, 0.86561445975026541 ]
# zoom=0.37599999999999989

# front=[ -0.75245044614233414, 0.02581722023130522, 0.65814268759876893 ]
# lookat=[ 0.43410814044790336, 2.7563779158502544, -5.4251798502022259 ]
# up=[ 0.61722127731746712, 0.37642582069926311, 0.69089904931001367 ]
# zoom=0.37599999999999989

front=[ -0.85489400745527955, -0.21258710380031173, 0.47324724966432491 ]
lookat=[ 0.56676915040846054, 3.0302380791392753, -4.9474472640869207 ]
up=[ 0.25964207431035369, 0.61442092913670088, 0.74503215708223625 ]
zoom=0.49599999999999989

# front=[ -0.11474575071162264, -0.0075117158831180721, 0.9933664916928282 ]
# lookat=[ 0.56676915040846054, 3.0302380791392753, -4.9474472640869207 ]
# up=[ 0.66533910330218726, 0.74197260227638662, 0.082465356897599429 ]
# zoom=0.13999999999999999

## Point cloud data preparation
DATANAME = "ITC_groundfloor.ply"
pcd = o3d.io.read_point_cloud("/Users/ctl/USTH/3DDeepLearning/DATA/" + DATANAME)
pcd_center = pcd.get_center()
pcd.translate(-pcd_center)
o3d.visualization.draw_geometries([pcd],"Original Point Cloud",zoom=zoom,front=front,lookat=lookat,up=up)

## Mean distance to nearest neighbor
nn_distance_cal = np.mean(pcd.compute_nearest_neighbor_distance())
print("Mean distance between points",nn_distance_cal)

## Point Cloud Reduction: Random Sampling
retained_ratio = 0.05 # Keep only 20%
sampled_pcd = pcd.random_down_sample(retained_ratio)
print(sampled_pcd)
# o3d.visualization.draw_geometries([sampled_pcd],"Random Sampling",zoom=zoom,front=front,lookat=lookat,up=up)

## Statistical outliers
nn = 18 #number of neighbors
std_multiplier = 10 #low=aggressive 
filtered_pcd, filtered_idx = pcd.remove_statistical_outlier(nn, std_multiplier)
outliers = pcd.select_by_index(filtered_idx, invert=True)
outliers.paint_uniform_color([1, 0, 0])
print("Number of outliers points:", len(outliers.points))
o3d.visualization.draw_geometries([filtered_pcd, outliers],"Outlier Removal",zoom=zoom,front=front,lookat=lookat,up=up)

#### Voxel Sub-sampling
voxel_size = 0.05
pcd_downsampled = filtered_pcd.voxel_down_sample(voxel_size = voxel_size)
print(pcd_downsampled)
o3d.visualization.draw_geometries([pcd_downsampled],"Voxel Sub-sampling",zoom=zoom,front=front,lookat=lookat,up=up)

#### Normal Extraction
nn_distance = 0.05
radius_normals=nn_distance*4
pcd_downsampled.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normals, max_nn=16), fast_normal_computation=True)
pcd_downsampled.paint_uniform_color([0.6, 0.6, 0.6])
o3d.visualization.draw_geometries([pcd_downsampled],"Normal Extraction",zoom=zoom,front=front,lookat=lookat,up=up)

#### RANSAC First-order
pcd_ransac_single=pcd_downsampled
distance_threshold = 0.1 #Larget distance point to plane to be segmented
ransac_n = 3 # Inital point to interate
num_iterations = 1000
plane_model, inliers = pcd_ransac_single.segment_plane(distance_threshold=distance_threshold,ransac_n=3,num_iterations=1000)
[a, b, c, d] = plane_model
print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

inlier_cloud = pcd_ransac_single.select_by_index(inliers) #Plane
outlier_cloud = pcd_ransac_single.select_by_index(inliers, invert=True) #Not the plane
inlier_cloud.paint_uniform_color([1.0, 0, 0]) 
outlier_cloud.paint_uniform_color([0.6, 0.6, 0.6])
o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],"RANSAC Single-order",zoom=zoom,front=front,lookat=lookat,up=up)

#### RANSAC Multi-order
pcd_ransac_multi=pcd_downsampled
segment_models={} #Plane parameter  a,b,c,d
segments={} #Planer region (point in that region)
max_plane_idx=10 #Assume that only have 10 total surface

# for i in range(max_plane_idx):
#     segment_models[i], inliers = pcd_ransac_multi.segment_plane(distance_threshold=distance_threshold,ransac_n=3,num_iterations=1000)
#     #Collecting segmentation
#     segments[i]=pcd_ransac_multi.select_by_index(inliers)
#     #Continute with outlier
#     pcd_ransac_multi = pcd_ransac_multi.select_by_index(inliers, invert=True)
#     #Visualization
#     colors = plt.get_cmap("tab20")(i)
#     segments[i].paint_uniform_color(list(colors[:3]))
#     print("pass",i+1,"/",max_plane_idx,"done.")

# o3d.visualization.draw_geometries([segments[i] for i in range(max_plane_idx)]+[pcd_ransac_multi],"RANSAC Multi-order",zoom=zoom,front=front,lookat=lookat,up=up)


#### DBSCAN
epsilon = 0.15
min_cluster_points = 5
# # Run on top of RANSAC
for i in range(max_plane_idx):
    segment_models[i], inliers = pcd_ransac_multi.segment_plane(distance_threshold=distance_threshold,ransac_n=3,num_iterations=1000)
    #Collecting segmentation
    segments[i]=pcd_ransac_multi.select_by_index(inliers)
    #Clustering again the inlier
    labels = np.array(segments[i].cluster_dbscan(eps=epsilon, min_points=min_cluster_points))
    #Count point in each cluster of the given inlier
    candidates=[len(np.where(labels==j)[0]) for j in np.unique(labels)]
    #Choose best candidate to represent the inlier
    best_candidate=int(np.unique(labels)[np.where(candidates== np.max(candidates))[0]])
    # Continue with outlier + clustered inliers
    pcd_ransac_multi = pcd_ransac_multi.select_by_index(inliers, invert=True) + segments[i].select_by_index(list(np.where(labels!=best_candidate)[0]))
    segments[i]=segments[i].select_by_index(list(np.where(labels== best_candidate)[0]))
    #Visualization
    colors = plt.get_cmap("tab20")(i)
    segments[i].paint_uniform_color(list(colors[:3]))
    print("pass",i+1,"/",max_plane_idx,"done.")

o3d.visualization.draw_geometries([segments[i] for i in range(max_plane_idx)]+[pcd_ransac_multi],"RANSAC with DBSCAN",zoom=zoom,front=front,lookat=lookat,up=up)

#### DBSCAN Refinement
o3d.visualization.draw_geometries([pcd_ransac_multi],"Rest not yet segmented",zoom=zoom,front=front,lookat=lookat,up=up)

labels = np.array(pcd_ransac_multi.cluster_dbscan(eps=0.1, min_points=1))
max_label = labels.max()
colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 0
pcd_ransac_multi.colors = o3d.utility.Vector3dVector(colors[:, :3])

num_noise_points = np.sum(labels == -1)
print(labels)
o3d.visualization.draw_geometries([pcd_ransac_multi],"Rest segmented",zoom=zoom,front=front,lookat=lookat,up=up)
o3d.visualization.draw_geometries([segments[i] for i in range(max_plane_idx)]+[pcd_ransac_multi],"Final RANSAC with DBSCAN",zoom=zoom,front=front,lookat=lookat,up=up)


                  
# Plotting DBSCAN
# num_clusters_list = []
# num_noise_points_list = []
# for min_points in range(1, 11):
#     labels = np.array(pcd_ransac_multi.cluster_dbscan(eps=0.1, min_points=min_points))
#     num_clusters = len(np.unique(labels[labels != -1]))
#     num_noise_points = np.sum(labels == -1)
#     num_clusters_list.append(num_clusters)
#     num_noise_points_list.append(num_noise_points)

# plt.figure(figsize=(10, 6))
# plt.plot(range(1, 11), num_clusters_list, marker='o', label='Number of Clusters')
# plt.plot(range(1, 11), num_noise_points_list, marker='o', label='Number of Noise Points')
# plt.title('Clustering Performance vs. min_points')
# plt.xlabel('min_points')
# plt.ylabel('Count')
# plt.legend()
# plt.grid(True)
# plt.xticks(range(1, 11))
# plt.show()

# Read .ply file
# input_file = "/Users/ctl/USTH/3DDeepLearning/DATA/ITC_groundfloor.ply"
# pcd = o3d.io.read_point_cloud(input_file) # Read the point cloud
# # Convert open3d format to numpy array
# point_cloud_in_numpy = np.asarray(pcd.points) 
# print(pcd)
# print(point_cloud_in_numpy)
