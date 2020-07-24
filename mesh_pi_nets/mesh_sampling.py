### Code obtained and modified from https://github.com/anuragranj/coma, Copyright (c) 2018 Anurag Ranjan, Timo Bolkart, Soubhik Sanyal, Michael J. Black and the Max Planck Gesellschaft

import math
import heapq
import numpy as np
import os
import scipy.sparse as sp
from scipy import spatial
import pickle
import trimesh

try:
    import psbody.mesh

    found = True
except ImportError:
    found = False
if found:
    from psbody.mesh import Mesh


def compute_downsampling(downsample_directory, downsample_method, shapedata, ds_factors):
    if not os.path.exists(os.path.join(downsample_directory, 'downsampling_matrices.pkl')):
        if shapedata.meshpackage == 'trimesh':
            raise NotImplementedError

        print("Generating Transform Matrices ..")
        if downsample_method == 'COMA_downsample':
            M, A, D, U, F = generate_transform_matrices(shapedata.reference_mesh, ds_factors)
        else:
            raise NotImplementedError(downsample_method)

        with open(os.path.join(downsample_directory, 'downsampling_matrices.pkl'), 'wb') as fp:
            M_verts_faces = [(M[i].v, M[i].f) for i in range(len(M))]
            pickle.dump({'M_verts_faces': M_verts_faces, 'A': A, 'D': D, 'U': U, 'F': F}, fp)
    else:
        print("Loading Transform Matrices ..")
        with open(os.path.join(downsample_directory, 'downsampling_matrices.pkl'), 'rb') as fp:
            downsampling_matrices = pickle.load(fp, encoding='latin1')

        M_verts_faces = downsampling_matrices['M_verts_faces']
        if shapedata.meshpackage == 'mpi-mesh':
            M = [Mesh(v=M_verts_faces[i][0], f=M_verts_faces[i][1]) for i in range(len(M_verts_faces))]
        elif shapedata.meshpackage == 'trimesh':
            M = [trimesh.base.Trimesh(vertices=M_verts_faces[i][0], faces=M_verts_faces[i][1], process=False)
                 for i in range(len(M_verts_faces))]
        A = downsampling_matrices['A']
        D = downsampling_matrices['D']
        U = downsampling_matrices['U']
        F = downsampling_matrices['F']

    return M, A, D, U, F


def vertex_quadrics(mesh):
    """Computes a quadric for each vertex in the Mesh.

    Returns:
       v_quadrics: an (N x 4 x 4) array, where N is # vertices.
    """

    # Allocate quadrics
    v_quadrics = np.zeros((len(mesh.v), 4, 4,))

    # For each face...
    for f_idx in range(len(mesh.f)):

        # Compute normalized plane equation for that face
        vert_idxs = mesh.f[f_idx]
        verts = np.hstack((mesh.v[vert_idxs], np.array([1, 1, 1]).reshape(-1, 1)))
        u, s, v = np.linalg.svd(verts)
        eq = v[-1, :].reshape(-1, 1)
        eq = eq / (np.linalg.norm(eq[0:3]))

        # Add the outer product of the plane equation to the
        # quadrics of the vertices for this face
        for k in range(3):
            v_quadrics[mesh.f[f_idx, k], :, :] += np.outer(eq, eq)

    return v_quadrics


def setup_deformation_transfer(source, target, use_normals=False):
    rows = np.zeros(3 * target.v.shape[0])
    cols = np.zeros(3 * target.v.shape[0])
    coeffs_v = np.zeros(3 * target.v.shape[0])
    coeffs_n = np.zeros(3 * target.v.shape[0])

    nearest_faces, nearest_parts, nearest_vertices = source.compute_aabb_tree().nearest(target.v, True)
    nearest_faces = nearest_faces.ravel().astype(np.int64)
    nearest_parts = nearest_parts.ravel().astype(np.int64)
    nearest_vertices = nearest_vertices.ravel()

    for i in range(target.v.shape[0]):
        # Closest triangle index
        f_id = nearest_faces[i]
        # Closest triangle vertex ids
        nearest_f = source.f[f_id]

        # Closest surface point
        nearest_v = nearest_vertices[3 * i:3 * i + 3]
        # Distance vector to the closest surface point
        dist_vec = target.v[i] - nearest_v

        rows[3 * i:3 * i + 3] = i * np.ones(3)
        cols[3 * i:3 * i + 3] = nearest_f

        n_id = nearest_parts[i]
        if n_id == 0:
            # Closest surface point in triangle
            A = np.vstack((source.v[nearest_f])).T
            coeffs_v[3 * i:3 * i + 3] = np.linalg.lstsq(A, nearest_v)[0]
        elif n_id > 0 and n_id <= 3:
            # Closest surface point on edge
            A = np.vstack((source.v[nearest_f[n_id - 1]], source.v[nearest_f[n_id % 3]])).T
            tmp_coeffs = np.linalg.lstsq(A, target.v[i])[0]
            coeffs_v[3 * i + n_id - 1] = tmp_coeffs[0]
            coeffs_v[3 * i + n_id % 3] = tmp_coeffs[1]
        else:
            # Closest surface point a vertex
            coeffs_v[3 * i + n_id - 4] = 1.0

    #    if use_normals:
    #        A = np.vstack((vn[nearest_f])).T
    #        coeffs_n[3 * i:3 * i + 3] = np.linalg.lstsq(A, dist_vec)[0]

    # coeffs = np.hstack((coeffs_v, coeffs_n))
    # rows = np.hstack((rows, rows))
    # cols = np.hstack((cols, source.v.shape[0] + cols))
    matrix = sp.csc_matrix((coeffs_v, (rows, cols)), shape=(target.v.shape[0], source.v.shape[0]))
    return matrix


def qslim_decimator_transformer(mesh, factor=None, n_verts_desired=None):
    from opendr.topology import get_vertices_per_edge
    """Return a simplified version of this mesh.

    A Qslim-style approach is used here.

    :param factor: fraction of the original vertices to retain
    :param n_verts_desired: number of the original vertices to retain
    :returns: new_faces: An Fx3 array of faces, mtx: Transformation matrix
    """

    if factor is None and n_verts_desired is None:
        raise Exception('Need either factor or n_verts_desired.')

    if n_verts_desired is None:
        n_verts_desired = math.ceil(len(mesh.v) * factor)

    Qv = vertex_quadrics(mesh)

    # fill out a sparse matrix indicating vertex-vertex adjacency
    # from psbody.mesh.topology.connectivity import get_vertices_per_edge
    vert_adj = get_vertices_per_edge(mesh.v, mesh.f)
    # vert_adj = sp.lil_matrix((len(mesh.v), len(mesh.v)))
    # for f_idx in range(len(mesh.f)):
    #     vert_adj[mesh.f[f_idx], mesh.f[f_idx]] = 1

    vert_adj = sp.csc_matrix((vert_adj[:, 0] * 0 + 1, (vert_adj[:, 0], vert_adj[:, 1])),
                             shape=(len(mesh.v), len(mesh.v)))
    vert_adj = vert_adj + vert_adj.T
    vert_adj = vert_adj.tocoo()

    def collapse_cost(Qv, r, c, v):
        Qsum = Qv[r, :, :] + Qv[c, :, :]
        p1 = np.vstack((v[r].reshape(-1, 1), np.array([1]).reshape(-1, 1)))
        p2 = np.vstack((v[c].reshape(-1, 1), np.array([1]).reshape(-1, 1)))

        destroy_c_cost = p1.T.dot(Qsum).dot(p1)
        destroy_r_cost = p2.T.dot(Qsum).dot(p2)
        result = {
            'destroy_c_cost': destroy_c_cost,
            'destroy_r_cost': destroy_r_cost,
            'collapse_cost': min([destroy_c_cost, destroy_r_cost]),
            'Qsum': Qsum}
        return result

    # construct a queue of edges with costs
    queue = []
    for k in range(vert_adj.nnz):
        r = vert_adj.row[k]
        c = vert_adj.col[k]

        if r > c:
            continue

        cost = collapse_cost(Qv, r, c, mesh.v)['collapse_cost']
        heapq.heappush(queue, (cost, (r, c)))

    # decimate
    collapse_list = []
    nverts_total = len(mesh.v)
    faces = mesh.f.copy()
    while nverts_total > n_verts_desired:
        e = heapq.heappop(queue)
        r = e[1][0]
        c = e[1][1]
        if r == c:
            continue

        cost = collapse_cost(Qv, r, c, mesh.v)
        if cost['collapse_cost'] > e[0]:
            heapq.heappush(queue, (cost['collapse_cost'], e[1]))
            # print 'found outdated cost, %.2f < %.2f' % (e[0], cost['collapse_cost'])
            continue
        else:

            # update old vert idxs to new one,
            # in queue and in face list
            if cost['destroy_c_cost'] < cost['destroy_r_cost']:
                to_destroy = c
                to_keep = r
            else:
                to_destroy = r
                to_keep = c

            collapse_list.append([to_keep, to_destroy])

            # in our face array, replace "to_destroy" vertidx with "to_keep" vertidx
            np.place(faces, faces == to_destroy, to_keep)

            # same for queue
            which1 = [idx for idx in range(len(queue)) if queue[idx][1][0] == to_destroy]
            which2 = [idx for idx in range(len(queue)) if queue[idx][1][1] == to_destroy]
            for k in which1:
                queue[k] = (queue[k][0], (to_keep, queue[k][1][1]))
            for k in which2:
                queue[k] = (queue[k][0], (queue[k][1][0], to_keep))

            Qv[r, :, :] = cost['Qsum']
            Qv[c, :, :] = cost['Qsum']

            a = faces[:, 0] == faces[:, 1]
            b = faces[:, 1] == faces[:, 2]
            c = faces[:, 2] == faces[:, 0]

            # remove degenerate faces
            def logical_or3(x, y, z):
                return np.logical_or(x, np.logical_or(y, z))

            faces_to_keep = np.logical_not(logical_or3(a, b, c))
            faces = faces[faces_to_keep, :].copy()

        nverts_total = (len(np.unique(faces.flatten())))

    new_faces, mtx = _get_sparse_transform(faces, len(mesh.v))
    return new_faces, mtx


def _get_sparse_transform(faces, num_original_verts):
    verts_left = np.unique(faces.flatten())
    IS = np.arange(len(verts_left))
    JS = verts_left
    data = np.ones(len(JS))

    mp = np.arange(0, np.max(faces.flatten()) + 1)
    mp[JS] = IS
    new_faces = mp[faces.copy().flatten()].reshape((-1, 3))

    ij = np.vstack((IS.flatten(), JS.flatten()))
    mtx = sp.csc_matrix((data, ij), shape=(len(verts_left), num_original_verts))

    return (new_faces, mtx)


def generate_transform_matrices(mesh, factors):
    from opendr.topology import get_vert_connectivity
    """Generates len(factors) meshes, each of them is scaled by factors[i] and
       computes the transformations between them.

    Returns:
       M: a set of meshes downsampled from mesh by a factor specified in factors.
       A: Adjacency matrix for each of the meshes
       D: Downsampling transforms between each of the meshes
       U: Upsampling transforms between each of the meshes
    """

    factors = map(lambda x: 1.0 / x, factors)
    M, A, D, U = [], [], [], []
    ## sergey code
    F = []
    ##
    A.append(get_vert_connectivity(mesh.v, mesh.f))
    M.append(mesh)

    i = 0
    for factor in factors:
        ds_f, ds_D = qslim_decimator_transformer(M[-1], factor=factor)
        D.append(ds_D)
        ##
        F.append(ds_f)
        ##
        new_mesh_v = ds_D.dot(M[-1].v)
        new_mesh = Mesh(v=new_mesh_v, f=ds_f)
        M.append(new_mesh)
        A.append(get_vert_connectivity(new_mesh.v, new_mesh.f))
        U.append(setup_deformation_transfer(M[-1], M[-2]))
        print('decimation %d by factor %.2f finished' % (i, factor))
        i += 1

    return M, A, D, U, F


def generate_transform_matrices_given_downsamples(mesh, downsample_directory, num_downsamples=4):
    from opendr.topology import get_vert_connectivity
    M, A, D, U, F = [], [], [], [], []
    A.append(get_vert_connectivity(mesh.v, mesh.f))
    M.append(mesh)

    for i in range(1, num_downsamples + 1):
        #         import trimesh
        #         cur_M = trimesh.load(os.path.join(downsample_directory,'template_d{0}.obj'.format(i)))
        cur_M = Mesh(filename=os.path.join(downsample_directory, 'template_d{0}.obj'.format(i)))
        cur_D = np.zeros((cur_M.v.shape[0], M[-1].v.shape[0]))
        kd = spatial.KDTree(np.array(M[-1].v))
        for vi in range(cur_M.v.shape[0]):
            _, u = kd.query(cur_M.v[vi])
            cur_D[vi, u] = 1.0
        M.append(cur_M)
        D.append(sp.csr_matrix(cur_D))
        F.append(cur_M.f)
        A.append(get_vert_connectivity(cur_M.v, cur_M.f))
        U.append(setup_deformation_transfer(M[-1], M[-2]))

    return M, A, D, U, F

