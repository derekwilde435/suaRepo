import numpy as np
from mat import mat
from utils import in_half_plane, s_norm, Rz, angle, i2p


class Algorithms:
    def __init__(self):
        self.i = 0
        self.state = 0

    def pathFollower(self, flag, r, q, p, chi, chi_inf, k_path, c, rho, lamb, k_orbit):
        """
        Input:
            flag = 1 for straight line, 2 for orbit
            r = origin of straight-line path in NED (m)
            q = direction of straight-line path in NED (m)
            p = current position of uav in NED (m)
            chi = course angle of UAV (rad)
            chi_inf = straight line path following parameter
            k_path = straight line path following parameter
            c = center of orbit in NED (m)
            rho = radius of orbit (m)
            lamb = direction of orbit, 1 clockwise, -1 counter-clockwise
            k_orbit = orbit path following parameter

        Outputs:
            e_crosstrack = crosstrack error (m)
            chi_c = commanded course angle (rad)
            h_c = commanded altitude (m)

        Example Usage
            e_crosstrack, chi_c, h_c = pathFollower(path)

        Reference: Beard, Small Unmanned Aircraft, Chapter 10, Algorithms 3 and 4
        Copyright 2018 Utah State University
        """

        # Unpackaging variables
        r_n = r[0][0]
        r_e = r[1][0]
        r_d = r[2][0]
        p_n = p[0][0]
        p_e = p[1][0]
        p_d = p[2][0]
        k_i = np.array([0, 0, 1])
        e_i_p = np.array([[p_n-r_n], [p_e-r_e], [p_d-r_d]])  # 3x1
        normal = (np.cross(q.T, k_i)/(np.linalg.norm(np.cross(q.T, k_i)))).T  # 3x1
        s_i = np.subtract(e_i_p, (np.dot(e_i_p.T, normal) * normal))  # 3x1
        s_n = s_i[0]
        s_e = s_i[1]
        # s_d = 
        q_n = q[0]
        q_e = q[1]
        q_d = q[2]
        chi_q = np.arctan2(q_e, q_n)
        chi = chi

        c_n = c[0][0]
        c_e = c[1][0]
        c_d = c[2][0]
        lamb = lamb
        k_orbit = 3.6
        rho = 55

        if flag == 1:  # straight line
            pass
            # TODO Algorithm 3 goes here
            h_d = -r_d + np.sqrt(s_n**2 + s_e**2) * (q_d / np.sqrt(q_n**2 + q_e**2)) #equation 10.5
            chi_q = np.arctan2(q_e, q_n)
            while (chi_q - chi) < -np.pi:
                chi_q = chi_q + 2 * np.pi
            while (chi_q - chi) > np.pi:
                chi_q = chi_q - 2 * np.pi
            e_py = -np.sin(chi_q) * (p_n - r_n) + np.cos(chi_q) * (p_e - r_e)
            e_crosstrack = e_py
            chi_c = chi_q - chi_inf * 2 / np.pi * np.arctan(k_path * e_py)
            h_c = h_d
        elif flag == 2:  # orbit following
            pass
            # TODO Algorithm 4 goes here
            h_c = -c_d
            d = np.sqrt((p_n - c_n)**2 + (p_e - c_e)**2)
            psi = np.arctan2(p_e - c_e, p_n - c_n)
            while (psi - chi) < -np.pi:
                psi = psi + 2 * np.pi
            while (psi - chi) > np.pi:
                psi = psi - 2 * np.pi
            chi_c = 0.6 + psi + lamb * (np.pi / 2 + np.arctan(k_orbit * ((d - rho) / rho)))
            e_crosstrack = d - rho
        else:
            raise Exception("Invalid path type")

        return e_crosstrack, chi_c, h_c

    # followWpp algorithm left here for reference
    # It is not used in the final implementation
#     def followWpp(self, w, p, newpath):

        """
        followWpp implements waypoint following via connected straight-line
        paths.

        Inputs:
            w = 3xn matrix of waypoints in NED (m)
            p = position of MAV in NED (m)
            newpath = flag to initialize the algorithm or define new waypoints

        Outputs
            r = origin of straight-line path in NED (m)
            q = direction of straight-line path in NED (m)

        Example Usage;
            r, q = followWpp(w, p, newpath)

        Reference: Beard, Small Unmanned Aircraft, Chapter 11, Algorithm 5
        Copyright 2018 Utah State University
        """

        if self.i is None:
            self.i = 0

        if newpath:
            # initialize index
            self.i = 1

        # check sizes
        m, N = w.shape
        assert N >= 3
        assert m == 3

        # calculate the q vector
        r = w[:, self.i - 1]
        qi1 = s_norm(w[:, self.i], -w[:, self.i - 1])

        # Calculate the origin of the current path
        qi = s_norm(w[:, self.i + 1], -w[:, self.i])

        # Calculate the unit normal to define the half plane
        ni = s_norm(qi1, qi)

        # Check if the MAV has crossed the half-plane
        if in_half_plane(p, w[:, self.i], ni):
            if self.i < (N - 2):
                self.i += 1
        q = qi1
        return r, q

    # followWppFillet algorithm left here for reference.
    # It is not used in the final implementation
    def followWppFillet(self, w, p, R, newpath):
        """
        followWppFillet implements waypoint following via straightline paths
        connected by fillets

        Inputs:
            W = 3xn matrix of waypoints in NED (m)
            p = position of MAV in NED (m)
            R = fillet radius (m)
            newpath = flag to initialize the algorithm or define new waypoints

        Outputs
            flag = flag for straight line path (1) or orbit (2)
            r = origin of straight-line path in NED (m)
            q = direction of straight-line path in NED (m)
            c = center of orbit in NED (m)
            rho = radius of orbit (m)
            lamb = direction or orbit, 1 clockwise, -1 counter clockwise

        Example Usage
            [flag, r, q, c, rho, lamb] = followWppFillet( w, p, R, newpath )

        Reference: Beard, Small Unmanned Aircraft, Chapter 11, Algorithm 6
        Copyright 2018 Utah State University
        """

        if self.i is None:
            self.i = 0
            self.state = 0
        if newpath:
            # Initialize the waypoint index
            self.i = 2
            self.state = 1

            # Check size of waypoints matrix
            m, N = w.shape  # Where 'N' is the number of waypoints and 'm' dimensions
            assert N >= 3
            assert m == 3
        else:
            [m, N] = w.shape
            assert N >= 3
            assert m == 3
        # Calculate the q vector and fillet angle
        qi1 = mat(s_norm(w[:, self.i], -w[:, self.i - 1]))
        qi = mat(s_norm(w[:, self.i + 1], -w[:, self.i]))
        e = acos(-qi1.T * qi)

        # Determine if the MAV is on a straight or orbit path
        if self.state == 1:
            c = mat([0, 0, 0]).T
            rho = 0
            lamb = 0

            flag = 1
            r = w[:, self.i - 1]
            q = q1
            z = w[:, self.i] - (R / (np.tan(e / 2))) * qi1
            if in_half_plane(p, z, qi1):
                self.state = 2

        elif self.state == 2:
            r = [0, 0, 0]
            q = [0, 0, 0]

            flag = 2
            c = w[:, self.i] - (R / (np.sin(e / 2))) * s_norm(qi1, -qi)
            rho = R
            lamb = np.sign(qi1(1) * qi(2) - qi1(2) * qi(1))
            z = w[:, self.i] + (R / (np.tan(e / 2))) * qi

            if in_half_plane(p, z, qi):
                if self.i < (N - 1):
                    self.i = self.i + 1
                self.state = 1

        else:
            # Fly north as default
            flag = -1
            r = p
            q = mat([1, 0, 0]).T
            c = np.nan(3, 1)
            rho = np.nan
            lamb = np.nan

        return flag, r, q, c, rho, lamb

    def findDubinsParameters(self, p_s, chi_s, p_e, chi_e, R):
        """
        findDubinsParameters determines the dubins path parameters

        Inputs:
        p_s = start position (m)
        chi_s = start course angle (rad)
        p_e = end position (m)
        chi_e = end course angle (rad)
        R = turn radius (m)

        Outputs
        dp.L = path length (m)
        dp.c_s = start circle origin (m)
        dp.lamb_s = start circle direction (unitless)
        dp.c_e = end circle origin (m)
        dp.lamb_e = end circle direction (unitless)
        dp.z_1 = Half-plane H_1 location (m)
        dp.q_12 = Half-planes H_1 and H_2 unit normals (unitless)
        dp.z_2 = Half-plane H_2 location (m)
        dp.z_3 = Half-plane H_3 location  (m)
        dp.q_3 = Half-plane H_3 unit normal (m)
        dp.case = case (unitless)

        Example Usage
        dp = findDubinsParameters( p_s, chi_s, p_e, chi_e, R )

        Reference: Beard, Small Unmanned Aircraft, Chapter 11, Algorithm 7
        Copyright 2018 Utah State University
        """

        # TODO Algorithm 7 goes here
        # Ensure the configurations are far enough apart
        assert(np.linalg.norm(p_s - p_e) >= 3*R, 'Start and end configurations are too close!')
        # Misc calcs
        e_1 = np.array([[1], [0], [0]])
        # Compute circle centers
        c_chi_s = np.cos(chi_s)
        s_chi_s = np.sin(chi_s)
        c_chi_e = np.cos(chi_e)
        s_chi_e = np.sin(chi_e)
        c_rs = p_s + R * Rz(np.pi / 2) * np.array([[c_chi_s], [s_chi_s], [0]])
        c_ls = p_s + R * Rz(-np.pi / 2) * np.array([[c_chi_s], [s_chi_s], [0]])
        c_re = p_e + R * Rz(np.pi / 2) * np.array([[c_chi_e], [s_chi_e], [0]])
        c_le = p_e + R * Rz(-np.pi / 2) * np.array([[c_chi_e], [s_chi_e], [0]])
        # Compute path lengths
        # Case 1: R-S-R
        th = angle(c_re - c_rs)
        L1 = s_norm(c_rs - c_re) \
            + R*((2*np.pi + ((th - np.pi/2) % (2*np.pi)) - ((chi_s - np.pi/2) % (2*np.pi))) % (2*np.pi))\
            + R*((2*np.pi + ((chi_e - np.pi/2) % (2*np.pi)) - ((th - np.pi/2) % (2*np.pi))) % (2*np.pi))
        # Case 2: R-S-L
        th = angle(c_le - c_rs)
        ell = s_norm(c_le - c_rs)
        th2 = th - np.pi/2 + np.arcsin(2*R/ell)
        if np.isreal(th2) != 1:
            L2 = np.nan  # will not be selected
        else:
            L2 = np.sqrt(ell**2 - 4*R**2) \
                + R*((2*np.pi + ((th2) % (2*np.pi)) - ((chi_s - np.pi/2) % (2*np.pi))) % (2*np.pi))\
                + R*((2*np.pi + ((th2 + np.pi) % (2*np.pi)) - ((chi_e + np.pi/2) % (2*np.pi))) % (2*np.pi))
        # Case 3: L-S-R
        th = angle(c_re - c_ls)
        ell = s_norm(c_re - c_ls)
        th2 = np.arccos(2 * R / ell)
        if np.isreal(th2) != 1:
            L3 = np.nan  # will not be selected
        else:
            L3 = np.sqrt(ell**2 - 4*R**2) \
                + R*((2*np.pi + ((chi_s + np.pi/2) % (2*np.pi)) - ((th + th2) % (2*np.pi))) % (2*np.pi))\
                + R*((2*np.pi + ((chi_e - np.pi/2) % (2*np.pi)) - ((th + th2 - np.pi) % (2*np.pi))) % (2*np.pi))
        # Case 4: L-S-L
        th = angle(c_le - c_ls)
        L4 = s_norm(c_ls - c_le) \
            + R*((2*np.pi + ((chi_s + np.pi/2) % (2*np.pi)) - ((th + np.pi/2) % (2*np.pi))) % (2*np.pi))\
            + R*((2*np.pi + ((th + np.pi/2) % (2*np.pi)) - ((chi_e + np.pi/2) % (2*np.pi))) % (2*np.pi))
        # Define the parameters for the minimum length path (ie: Dubins path)
        
        
        # package output into DubinsParameters class
        dp = DubinsParameters()

        # TODO populate dp members here

        return dp

    def followWppDubins(self, W, Chi, p, R, newpath):
        """
        followWppDubins implements waypoint following via Dubins paths

        Inputs:
            W = list of waypoints in NED (m)
            Chi = list of course angles at waypoints in NED (rad)
            p = mav position in NED (m)
            R = fillet radius (m)
            newpath = flag to initialize the algorithm or define new waypoints

        Outputs
            flag = flag for straight line path (1) or orbit (2)
            r = origin of straight-line path in NED (m)
            q = direction of straight-line path in NED (m)
            c = center of orbit in NED (m)
            rho = radius of orbit (m)
            lamb = direction or orbit, 1 clockwise, -1 counter clockwise
            self.i = waypoint number
            dp = dubins path parameters

        Example Usage
            flag, r, q, c, rho, lamb = followWppDubins(W, Chi, p, R, newpath)

        Reference: Beard, Small Unmanned Aircraft, Chapter 11, Algorithm 8
        Copyright 2018 Utah State University
        """

        # TODO Algorithm 8 goes here

        return flag, r, q, c, rho, lamb, self.i, dp


class DubinsParameters:
    def __init__(self):
        """
        Member Variables:
            L = path length (m)
            c_s = start circle origin (m)
            lamb_s = start circle direction (unitless)
            c_e = end circle origin (m)
            lamb_e = end circle direction (unitless)
            z_1 = Half-plane H_1 location (m)
            q_1 = Half-planes H_1 and H_2 unit normals (unitless)
            z_2 = Half-plane H_2 location (m)
            z_3 = Half-plane H_3 location  (m)
            q_3 = Half-plane H_3 unit normal (m)
            case = case (unitless)
        """

        self.L = 0
        self.c_s = mat([0, 0, 0]).T
        self.lamb_s = -1
        self.c_e = mat([0, 0, 0]).T
        self.lamb_e = 1
        self.z_1 = mat([0, 0, 0]).T
        self.q_1 = mat([0, 0, 0]).T
        self.z_2 = mat([0, 0, 0]).T
        self.z_3 = mat([0, 0, 0]).T
        self.q_3 = mat([0, 0, 0]).T
        self.case = 0
        self.lengths = np.array([[0, 0, 0, 0]])
        self.theta = 0
        self.ell = 0
        self.c_rs = mat([0, 0, 0]).T
        self.c_ls = mat([0, 0, 0]).T
        self.c_re = mat([0, 0, 0]).T
        self.c_le = mat([0, 0, 0]).T


