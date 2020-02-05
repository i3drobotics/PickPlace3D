import numpy as np
import math

_EPS = np.finfo(float).eps * 4.0

class Conversions():

    @staticmethod
    def d2r(d):
        return(d * math.pi/180)

    @staticmethod
    def r2d(r):
        return(r * 180/math.pi)

    @staticmethod
    def rpy2rv(roll,pitch,yaw):
        alpha = yaw
        beta = pitch
        gamma = roll
        
        ca = math.cos(alpha)
        cb = math.cos(beta)
        cg = math.cos(gamma)
        sa = math.sin(alpha)
        sb = math.sin(beta)
        sg = math.sin(gamma)
        
        r11 = ca*cb
        r12 = ca*sb*sg-sa*cg
        r13 = ca*sb*cg+sa*sg
        r21 = sa*cb
        r22 = sa*sb*sg+ca*cg
        r23 = sa*sb*cg-ca*sg
        r31 = -sb
        r32 = cb*sg
        r33 = cb*cg
        
        theta = math.acos((r11+r22+r33-1)/2)
        sth = math.sin(theta)
        kx = (r32-r23)/(2*sth)
        ky = (r13-r31)/(2*sth)
        kz = (r21-r12)/(2*sth)
        
        return (theta*kx),(theta*ky),(theta*kz)

    '''
    def rv2rpy(rx,ry,rz):
    
    theta = sqrt(rx*rx + ry*ry + rz*rz)
    kx = rx/theta
    ky = ry/theta
    kz = rz/theta
    cth = cos(theta)
    sth = sin(theta)
    vth = 1-cos(theta)
    
    r11 = kx*kx*vth + cth
    r12 = kx*ky*vth - kz*sth
    r13 = kx*kz*vth + ky*sth
    r21 = kx*ky*vth + kz*sth
    r22 = ky*ky*vth + cth
    r23 = ky*kz*vth - kx*sth
    r31 = kx*kz*vth - ky*sth
    r32 = ky*kz*vth + kx*sth
    r33 = kz*kz*vth + cth
    
    beta = atan2(-r31,sqrt(r11*r11+r21*r21))
    
    if beta > d2r(89.99):
        beta = d2r(89.99)
        alpha = 0
        gamma = atan2(r12,r22)
    elif beta < -d2r(89.99):
        beta = -d2r(89.99)
        alpha = 0
        gamma = -atan2(r12,r22)
    else:
        cb = cos(beta)
        alpha = atan2(r21/cb,r11/cb)
        gamma = atan2(r32/cb,r33/cb)
    
    return [r2d(gamma),r2d(beta),r2d(alpha)]
    '''
    
    @staticmethod
    def euler_to_quaternion(roll, pitch, yaw):
        qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
        qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
        qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

        return [qx, qy, qz, qw]

    @staticmethod
    def quaternion_to_euler(z, y, x, w):
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(t0, t1)
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch = math.asin(t2)
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(t3, t4)
        return [yaw, pitch, roll]

    @staticmethod
    def trig(angle):
        r = math.radians(angle)
        return math.cos(r), math.sin(r)

    @staticmethod
    def quaternion_from_transformation_matrix(matrix, isprecise=False):
        """Return quaternion from rotation matrix.

        If isprecise is True, the input matrix is assumed to be a precise rotation
        matrix and a faster algorithm is used.

        >>> q = quaternion_from_matrix(numpy.identity(4), True)
        >>> numpy.allclose(q, [1, 0, 0, 0])
        True
        >>> q = quaternion_from_matrix(numpy.diag([1, -1, -1, 1]))
        >>> numpy.allclose(q, [0, 1, 0, 0]) or numpy.allclose(q, [0, -1, 0, 0])
        True
        >>> R = rotation_matrix(0.123, (1, 2, 3))
        >>> q = quaternion_from_matrix(R, True)
        >>> numpy.allclose(q, [0.9981095, 0.0164262, 0.0328524, 0.0492786])
        True
        >>> R = [[-0.545, 0.797, 0.260, 0], [0.733, 0.603, -0.313, 0],
        ...      [-0.407, 0.021, -0.913, 0], [0, 0, 0, 1]]
        >>> q = quaternion_from_matrix(R)
        >>> numpy.allclose(q, [0.19069, 0.43736, 0.87485, -0.083611])
        True
        >>> R = [[0.395, 0.362, 0.843, 0], [-0.626, 0.796, -0.056, 0],
        ...      [-0.677, -0.498, 0.529, 0], [0, 0, 0, 1]]
        >>> q = quaternion_from_matrix(R)
        >>> numpy.allclose(q, [0.82336615, -0.13610694, 0.46344705, -0.29792603])
        True
        >>> R = random_rotation_matrix()
        >>> q = quaternion_from_matrix(R)
        >>> is_same_transform(R, quaternion_matrix(q))
        True
        >>> is_same_quaternion(quaternion_from_matrix(R, isprecise=False),
        ...                    quaternion_from_matrix(R, isprecise=True))
        True
        >>> R = euler_matrix(0.0, 0.0, numpy.pi/2.0)
        >>> is_same_quaternion(quaternion_from_matrix(R, isprecise=False),
        ...                    quaternion_from_matrix(R, isprecise=True))
        True

        """
        M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
        if isprecise:
            q = np.empty((4, ))
            t = np.trace(M)
            if t > M[3, 3]:
                q[0] = t
                q[3] = M[1, 0] - M[0, 1]
                q[2] = M[0, 2] - M[2, 0]
                q[1] = M[2, 1] - M[1, 2]
            else:
                i, j, k = 0, 1, 2
                if M[1, 1] > M[0, 0]:
                    i, j, k = 1, 2, 0
                if M[2, 2] > M[i, i]:
                    i, j, k = 2, 0, 1
                t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
                q[i] = t
                q[j] = M[i, j] + M[j, i]
                q[k] = M[k, i] + M[i, k]
                q[3] = M[k, j] - M[j, k]
                q = q[[3, 0, 1, 2]]
            q *= 0.5 / math.sqrt(t * M[3, 3])
        else:
            m00 = M[0, 0]
            m01 = M[0, 1]
            m02 = M[0, 2]
            m10 = M[1, 0]
            m11 = M[1, 1]
            m12 = M[1, 2]
            m20 = M[2, 0]
            m21 = M[2, 1]
            m22 = M[2, 2]
            # symmetric matrix K
            K = np.array([[m00-m11-m22, 0.0,         0.0,         0.0],
                            [m01+m10,     m11-m00-m22, 0.0,         0.0],
                            [m02+m20,     m12+m21,     m22-m00-m11, 0.0],
                            [m21-m12,     m02-m20,     m10-m01,     m00+m11+m22]])
            K /= 3.0
            # quaternion is eigenvector of K that corresponds to largest eigenvalue
            w, V = np.linalg.eigh(K)
            q = V[[3, 0, 1, 2], np.argmax(w)]
        if q[0] < 0.0:
            np.negative(q, q)
        return q

    @staticmethod
    def quaternion_to_transformation_matrix(quaternion):
        """Return homogeneous rotation matrix from quaternion.

        >>> M = quaternion_matrix([0.99810947, 0.06146124, 0, 0])
        >>> numpy.allclose(M, rotation_matrix(0.123, [1, 0, 0]))
        True
        >>> M = quaternion_matrix([1, 0, 0, 0])
        >>> numpy.allclose(M, numpy.identity(4))
        True
        >>> M = quaternion_matrix([0, 1, 0, 0])
        >>> numpy.allclose(M, numpy.diag([1, -1, -1, 1]))
        True

        """
        q = np.array(quaternion, dtype=np.float64, copy=True)
        n = np.dot(q, q)
        if n < _EPS:
            return np.identity(4)
        q *= math.sqrt(2.0 / n)
        q = np.outer(q, q)
        return np.array([
            [1.0-q[2, 2]-q[3, 3],     q[1, 2]-q[3, 0],     q[1, 3]+q[2, 0], 0.0],
            [    q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3],     q[2, 3]-q[1, 0], 0.0],
            [    q[1, 3]-q[2, 0],     q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2], 0.0],
            [                0.0,                 0.0,                 0.0, 1.0]])
    
    @staticmethod
    def euler_translation_to_transformation_matrix(rotation, translation):
        xC, xS = Conversions.trig(rotation[0])
        yC, yS = Conversions.trig(rotation[1])
        zC, zS = Conversions.trig(rotation[2])
        dX = translation[0]
        dY = translation[1]
        dZ = translation[2]
        Translate_matrix = np.array([[1, 0, 0, dX],
                                    [0, 1, 0, dY],
                                    [0, 0, 1, dZ],
                                    [0, 0, 0, 1]])
        Rotate_X_matrix = np.array([[1, 0, 0, 0],
                                    [0, xC, -xS, 0],
                                    [0, xS, xC, 0],
                                    [0, 0, 0, 1]])
        Rotate_Y_matrix = np.array([[yC, 0, yS, 0],
                                    [0, 1, 0, 0],
                                    [-yS, 0, yC, 0],
                                    [0, 0, 0, 1]])
        Rotate_Z_matrix = np.array([[zC, -zS, 0, 0],
                                    [zS, zC, 0, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]])
        return np.dot(Rotate_Z_matrix,np.dot(Rotate_Y_matrix,np.dot(Rotate_X_matrix,Translate_matrix)))