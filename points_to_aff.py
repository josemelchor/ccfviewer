import numpy as np

def points_to_aff(a, ab, ac):
    """ 
    Compute a 4x4 3D affine transform (and its inverse) from a point and 2 vectors defining a plane.  The transform maps:
        [0,0,0] => a
        [1,0,0] => b
        [0,1,0] => c
    The inverse matrix does exactly the opposite.

    Returns
    -------
    M, M_inverse
    """
    
    v0 = ab
    v1 = ac

    sv0 = np.linalg.norm(v0)
    sv1 = np.linalg.norm(v1)

    uv0 = v0 / sv0
    uv1 = v1 / sv1
    uv2 = np.cross(uv0, uv1)

    S = np.array([[ sv0, 0,   0, 0 ],
                  [ 0,   sv1, 0, 0 ],
                  [ 0,   0,   1, 0 ],
                  [ 0,   0,   0, 1 ]])

    R = np.array([ [ uv0[0], uv1[0], uv2[0], 0 ],
                   [ uv0[1], uv1[1], uv2[1], 0 ],
                   [ uv0[2], uv1[2], uv2[2], 0 ],
                   [ 0,      0,      0,      1 ] ])

    T = np.array([ [ 1, 0, 0, a[0] ],
                   [ 0, 1, 0, a[1] ],
                   [ 0, 0, 1, a[2] ],
                   [ 0, 0, 0, 1 ] ])

    M = np.dot(T, np.dot(R, S))

    return np.linalg.inv(M), M

def aff_to_lims_flat(M):
    """ flatten a 4x4 numpy array into a vector in the order expected by lims """
    return  [ M[0,0], M[0,1], M[0,2],
              M[1,0], M[1,1], M[1,2],
              M[2,0], M[2,1], M[2,2],
              M[0,3], M[1,3], M[2,3] ]

def lims_flat_to_aff(F):
    """ take a 1D vector in the order expected by lims and convert it into a 4x4 matrix """
    return np.array([ [ F[0], F[1], F[2], F[9]  ],
                      [ F[3], F[4], F[5], F[10] ],
                      [ F[6], F[7], F[8], F[11] ] ])

def aff_to_lims_obj(M, Mi):
    """ take a 4x4 matrix and its inverse and produce a dictionary with fields named as lims expects them """
    Mf = aff_to_lims_flat(M)
    Mif = aff_to_lims_flat(Mi)
    
    return {
        # tvr: transform volume (image coords) to reference (CCF coords)
        'tvr_00': Mif[0],
        'tvr_01': Mif[1],
        'tvr_02': Mif[2],
        'tvr_03': Mif[3],
        'tvr_04': Mif[4],
        'tvr_05': Mif[5],
        'tvr_06': Mif[6],
        'tvr_07': Mif[7],
        'tvr_08': Mif[8],
        'tvr_09': Mif[9],
        'tvr_10': Mif[10],
        'tvr_11': Mif[11],
        # trv: transform reference (CCF coords) to volume (image coords)
        'trv_00': Mf[0],
        'trv_01': Mf[1],
        'trv_02': Mf[2],
        'trv_03': Mf[3],
        'trv_04': Mf[4],
        'trv_05': Mf[5],
        'trv_06': Mf[6],
        'trv_07': Mf[7],
        'trv_08': Mf[8],
        'trv_09': Mf[9],
        'trv_10': Mf[10],
        'trv_11': Mf[11]
    }

def lims_obj_to_aff(ob):
    """ take a dictionary with input fields as lims expects them and output a 4x4 affine transform matrix and its inverse """
    
    M = lims_flat_to_aff([ob['trv_00'],
                          ob['trv_01'],
                          ob['trv_02'],
                          ob['trv_03'],
                          ob['trv_04'],
                          ob['trv_05'],
                          ob['trv_06'],
                          ob['trv_07'],
                          ob['trv_08'],
                          ob['trv_09'],
                          ob['trv_10'],
                          ob['trv_11']])

    Mi = lims_flat_to_aff([ob['tvr_00'],
                           ob['tvr_01'],
                           ob['tvr_02'],
                           ob['tvr_03'],
                           ob['tvr_04'],
                           ob['tvr_05'],
                           ob['tvr_06'],
                           ob['tvr_07'],
                           ob['tvr_08'],
                           ob['tvr_09'],
                           ob['tvr_10'],
                           ob['tvr_11']])

    return M, Mi

def aff_to_origin_and_vectors(M):
    """ use the 4x4 transform to compute the origin and vectors describing the plane expected by the ccf tool """
    a = np.dot(M, [ 0, 0, 0, 1 ])[:3]
    b = np.dot(M, [ 1, 0, 0, 1 ])[:3]
    c = np.dot(M, [ 0, 1, 0, 1 ])[:3]

    return a, b-a, c-a
#     
# def main():
#     # pick 20 random sets of points and try out the transform
#     N = 20
#     
#     for i in range(N):
#         # compute 3 random points
#         a = np.random.random(3)
#         b = np.random.random(3)
#         c = np.random.random(3)
# 
#         # compute vectors out of a
#         ab = b - a
#         ac = c - a
# 
#         # compute the 4x4 transform matrix
#         M0, M0i = points_to_aff(a, ab, ac)
# 
#         # build the lims dictionary
#         ob = aff_to_lims_obj(M0, M0i)
# 
#         # save it to a file if you want
#         # ...
# 
#         # convert it back into a matrix for testing
#         M1, M1i = lims_obj_to_aff(ob)
# 
#         a_new, ab_new, ac_new = aff_to_origin_and_vectors(M1i)
# 
#         print "***"
#         print "a before", a, "after", a_new, "diff", np.linalg.norm(a - a_new)
#         print "b before", ab, "after", ab_new, "diff", np.linalg.norm(ab - ab_new)
#         print "c before", ac, "after", ac_new, "diff", np.linalg.norm(ac - ac_new)
# 
# if __name__ == "__main__": main()

