import torch
import open3d as o3d
import numpy as np
from utils import FPS, nn_query, fmap2pointmap, read_shape
from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot as plt
import os
import trimesh
import argparse

def create_colormap(verts):
    minx = verts[:, 0].min()
    miny = verts[:, 1].min()
    minz = verts[:, 2].min()
    maxx = verts[:, 0].max()
    maxy = verts[:, 1].max()
    maxz = verts[:, 2].max()
    r = (verts[:, 0] - minx) / (maxx - minx)
    g = (verts[:, 1] - miny) / (maxy - miny)
    b = (verts[:, 2] - minz) / (maxz - minz)
    colors = np.stack((r, g, b), axis=-1)
    assert colors.shape == verts.shape
    return colors

def visualize_correspondence_single(p2p, vert1, vert2, point_size=10.0):
    vert1_offset = vert1 + np.array([1, 0, 0])
    all_points = np.vstack([vert1_offset, vert2])

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(all_points)

    colors = create_colormap(vert2)
    pc.colors = o3d.utility.Vector3dVector(np.vstack([colors[p2p], colors]))

    fps_sampler = FPS(vert1_offset, 50)
    fps_sampler.fit()
    selected_idx = fps_sampler.get_selected_idx()

    lines = [[idx, len(vert1) + p2p[idx]] for idx in selected_idx]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(all_points)
    line_set.lines = o3d.utility.Vector2iVector(np.array(lines))
    line_set.colors = o3d.utility.Vector3dVector(np.random.rand(len(lines), 3))

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    vis.add_geometry(pc)

    opt = vis.get_render_option()
    opt.point_size = point_size

    vis.run()
    vis.destroy_window()    

def compute_functional_map(basis1, descriptor1, basis2, descriptor2):
    basis1_trans = torch.inverse(basis1.T @ basis1) @ basis1.T
    coefficient1 = (basis1_trans @ descriptor1).T  

    basis2_trans = torch.inverse(basis2.T @ basis2) @ basis2.T
    coefficient2 = (basis2_trans @ descriptor2).T  

    C12 = torch.inverse(coefficient1.T @ coefficient1) @ coefficient1.T @ coefficient2
    
    return C12

def visualize_fmap(matrix, title="Functional Map Matrix C"):
    plt.imshow(matrix, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Basis 2")
    plt.ylabel("Basis 1")
    plt.show()

def process_and_visualize_single(embedding_path1, embedding_path2, descriptor_path1, descriptor_path2, shape_path1, shape_path2, use_fmap):
    emb1 = torch.from_numpy(torch.load(embedding_path1)).float()
    emb2 = torch.from_numpy(torch.load(embedding_path2)).float()

    if use_fmap:
        desc1 = torch.load(descriptor_path1).cpu()
        desc2 = torch.load(descriptor_path2).cpu()
    else:
        desc1, desc2 = None, None

    if use_fmap:
        C12 = compute_functional_map(basis1=emb1, basis2=emb2, descriptor1=desc1, descriptor2=desc2)
        visualize_fmap(C12.detach().numpy())
        p2p = fmap2pointmap(C12, emb2, emb1)
    else:
        p2p = nn_query(emb2, emb1)

    vert1 = torch.from_numpy(read_shape(shape_path1)[0]).float()
    vert2 = torch.from_numpy(read_shape(shape_path2)[0]).float()

    visualize_correspondence_single(p2p, vert1, vert2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and visualize shape correspondence")
    parser.add_argument('--embedding_path1', default="example/consistent_bases_mesh052.pt", help='Path to the first embedding file')
    parser.add_argument('--embedding_path2', default="example/consistent_bases_mesh053.pt", help='Path to the second embedding file')
    parser.add_argument('--descriptor_path1', default=None, help='Path to the first descriptor file')
    parser.add_argument('--descriptor_path2', default=None, help='Path to the second descriptor file')
    parser.add_argument('--shape_path1', default="example/mesh052.off", help='Path to the first shape file')
    parser.add_argument('--shape_path2', default="example/mesh053.off", help='Path to the second shape file')
    parser.add_argument('--use_fmap', action='store_true', help='Use functional map computation if set')

    args = parser.parse_args()

    process_and_visualize_single(
        embedding_path1=args.embedding_path1,
        embedding_path2=args.embedding_path2,
        descriptor_path1=args.descriptor_path1,
        descriptor_path2=args.descriptor_path2,
        shape_path1=args.shape_path1,
        shape_path2=args.shape_path2,
        use_fmap=args.use_fmap
    )
