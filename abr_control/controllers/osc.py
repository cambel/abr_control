import numpy as np

from abr_control.utils import transformations
from . import controller



class OSC(controller.Controller):
    """ Implements an operational space controller (OSC)

    Parameters
    ----------
    robot_config : class instance
        contains all relevant information about the arm
        such as: number of joints, number of links, mass information etc.
    kp : float, optional (Default: 1)
        proportional gain term on task space position
    kp : float, optional (Default: 1)
        proportional gain term on task space orientation
    kv : float, optional (Default: None)
        derivative gain term, a good starting point is sqrt(kp)
    ki : float, optional (Default: 0)
        integral gain term
    vmax : float, optional (Default: 0.5)
        The max allowed velocity of the end-effector [meters/second].
        If the control signal specifies something above this
        value it is clipped, if set to None no clipping occurs
    null_control : boolean, optional (Default: True)
        Apply a secondary control signal which
        drives the arm to specified resting joint angles without
        affecting the movement of the end-effector
    use_g : boolean, optional (Default: True)
        calculate and compensate for the effects of gravity
    use_C : boolean, optional (Default: False)
        calculate and compensate for the Coriolis and
        centripetal effects of the arm
    use_dJ : boolean, optional (Default: False)
        use the Jacobian derivative wrt time

    Attributes
    ----------
    nkv : float
        derivative gain term for null controller
    integrated_error : float list, optional (Default: None)
        task-space integrated error term
    """
    def __init__(self, robot_config, kp=1, ko=1, kv=None, ki=0, vmax=None,
                 null_control=True, use_g=True, use_C=False, use_dJ=False):

        super(OSC, self).__init__(robot_config)

        self.kp = kp
        self.ko = ko
        self.kv = np.sqrt(self.kp) if kv is None else kv
        self.ki = ki
        self.vmax = vmax
        self.lamb = self.kp / self.kv
        self.null_control = null_control
        self.use_g = use_g
        self.use_C = use_C
        self.use_dJ = use_dJ

        self.integrated_error = np.array([0.0, 0.0, 0.0])

        self.IDENTITY_N_JOINTS = np.eye(self.robot_config.N_JOINTS)
        self.ZEROS_THREE = np.zeros(3)
        # null space filter gains
        self.nkv = 10.0

    def generate(self, q, dq,
                 target_pos, target_vel=0,
                 ctrlr_dof=[True, True, True, False, False, False],
                 ref_frame='EE', offset=[0, 0, 0], ee_force=None):
        """ Generates the control signal to move the EE to a target

        Parameters
        ----------
        q : float numpy.array
            current joint angles [radians]
        dq : float numpy.array
            current joint velocities [radians/second]
        target_pos : float numpy.array
            desired task space position and orientation [meters, radians]
        target_vel : float numpy.array, optional (Default: 0)
            desired task space velocities [meters/sec, radians/sec]
        ctrlr_dof : list of boolean, optional (Default: position control)
            specifies which task space degrees of freedom are to be controlled
            [x, y, z, alpha, beta, gamma] (Default: [x, y, z]
            NOTE: if more ctrlr_dof are specified than degrees of freedom in
            the robotic system, the controller will perform poorly
        ref_frame : string, optional (Default: 'EE')
            the point being controlled, default is the end-effector.
        offset : list, optional (Default: [0, 0, 0])
            position offset inside the frame of reference [meters]
        """

        n_ctrlr_dof = np.sum(ctrlr_dof)

        print('target_pos: ', target_pos)

        new_target_pos = np.zeros(6)
        new_target_pos[ctrlr_dof] = target_pos
        target_pos = new_target_pos

        # calculate the end-effector position information
        xyz = self.robot_config.Tx(ref_frame, q, x=offset)

        # calculate the Jacobian for the end effector
        J = self.robot_config.J(ref_frame, q, x=offset)
        # isolate rows of Jacobian corresponding to controlled task space DOF
        J = J[ctrlr_dof]

        # calculate the inertia matrix in joint space -------------------------
        M = self.robot_config.M(q)

        # calculate the inertia matrix in task space
        M_inv = np.linalg.inv(M)
        # calculate the Jacobian for end-effector with no offset
        Mx_inv = np.dot(J, np.dot(M_inv, J.T))
        if np.linalg.det(Mx_inv) != 0:
            # do the linalg inverse if matrix is non-singular
            # because it's faster and more accurate
            Mx = np.linalg.inv(Mx_inv)
        else:
            # using the rcond to set singular values < thresh to 0
            # singular values < (rcond * max(singular_values)) set to 0
            Mx = np.linalg.pinv(Mx_inv, rcond=.005)

        # calculate the desired task space forces -----------------------------
        u_task = np.zeros(6)

        # calculate position error if position is being controlled
        if np.sum(ctrlr_dof[:3]) > 0:
            # TODO: how to get out the position component?
            print('ctrlr_dof[:3] ', ctrlr_dof[:3])
            u_task[:3][ctrlr_dof[:3]] = np.array(xyz - target_pos[:3])[ctrlr_dof[:3]]

        # calculate orientation error if orientation is being controlled
        if np.sum(ctrlr_dof[3:]) > 0:

            # # NOTE: is this appropriate? Calculating the current end effector
            # # orientation angles and replacing the ones being controlled?
            #
            # # calculate Euler angles for current orientation
            # R_EE = self.robot_config.R('EE', q)
            # angles = np.array(transformations.euler_from_matrix(R_EE, axes='sxyz'))

            # NOTE: it seems to work about the same with zeros instead
            # need to test in VREP, use zeros and save computation if same
            angles = np.zeros(3)
            # replace current angles with target angles
            angles[ctrlr_dof[3:]] = target_pos[ctrlr_dof[3:]]

            # generate quaternion representing target orientation
            q_target = transformations.quaternion_from_euler(
                angles[0], angles[1], angles[2], axes='sxyz')

            # from (Yuan, 1988), given r = [r1, r2, r3]
            # r^x = [[0, -r3, r2], [r3, 0, -r1], [-r2, r1, 0]]
            q_target_matrix = np.array([
                [0.0, -q_target[2], q_target[1]],
                [q_target[2], 0.0, -q_target[0]],
                [-q_target[1], q_target[0], 0.0]])

            # get the quaternion for the end effector
            q_EE = transformations.quaternion_from_matrix(
                self.robot_config.R('EE', q))

            # calculate the difference between q_EE and q_target
            # from (Yuan, 1988)
            # dq = (w_d * [x, y, z] - w * [x_d, y_d, z_d] -
            #       [x_d, y_d, z_d]^x * [x, y, z])
            u_task[3:] = (q_target[0] * q_EE[1:] - q_EE[0] * q_target[1:] -
                          np.dot(q_target_matrix, q_EE[1:]))

        # isolate task space forces corresponding to controlled DOF
        u_task = u_task[ctrlr_dof]

        # implement velocity limiting -----------------------------------------
        # if self.vmax is not None:
        #     sat = self.vmax / (self.lamb * np.abs(x_tilde))
        #     if np.any(sat < 1):
        #         index = np.argmin(sat)
        #         unclipped = self.kp * x_tilde[index]
        #         clipped = self.kv * self.vmax * np.sign(x_tilde[index])
        #         scale = np.ones(3, dtype='float32') * clipped / unclipped
        #         scale[index] = 1
        #     else:
        #         scale = np.ones(3, dtype='float32')
        #
        #     dx = np.dot(J, dq)
        #     if target_vel is None:
        #         target_vel = np.zeros(CTRLR_DOF)
        #     u_task[:3] = -self.kv * (dx - target_vel -
        #                              np.clip(sat / scale, 0, 1) *
        #                              -self.lamb * scale * x_tilde)
        #     # low level signal set to zero
        #     u = 0.0
        # else:

        # implement velocity limiting -----------------------------------------
        if self.vmax is not None:
            sat = self.vmax / (self.lamb * np.abs(u_task))
            if np.any(sat < 1):
                index = np.argmin(sat)
                unclipped = self.kp * u_task[index]
                clipped = self.kv * self.vmax * np.sign(u_task[index])
                scale = np.ones(n_ctrlr_dof, dtype='float32') * clipped / unclipped
                scale[index] = 1
            else:
                scale = np.ones(3, dtype='float32')

            dx = np.dot(J, dq)
            u_task = -self.kv * (dx - target_vel -
                                 np.clip(sat / scale, 0, 1) *
                                 -self.lamb * scale * u_task)
            # low level signal set to zero
            u = 0.0
        else:
            # generate (x,y,z) force without velocity limiting)
            u_task *= -self.kp
            if np.all(target_vel == 0):
                # if the target velocity is zero, it's more accurate to
                # apply velocity compensation in joint space
                u = -self.kv * np.dot(M, dq)
            else:
                dx = np.dot(J, dq)
                # high level signal includes velocity compensation
                u_task -= self.kv * (dx - target_vel)
                u = 0.0

        if self.use_dJ:
            # add in estimate of current acceleration
            dJ = self.robot_config.dJ(ref_frame, q=q, dq=dq)
            # apply mask
            dJ = dJ[:3]
            u_task += np.dot(dJ, dq)

        if self.ki != 0:
            # add in the integrated error term
            self.integrated_error += x_tilde
            u_task -= self.ki * self.integrated_error

        # incorporate task space inertia matrix
        u += np.dot(J.T, np.dot(Mx, u_task))

        if self.use_C:
            # add in estimation of full centrifugal and Coriolis effects
            u -= np.dot(self.robot_config.C(q=q, dq=dq), dq)

        # store the current control signal u for training in case
        # dynamics adaptation signal is being used
        # NOTE: training signal should not include gravity compensation
        self.training_signal = np.copy(u)

        # cancel out effects of gravity
        if self.use_g:
            # add in gravity term in joint space
            u -= self.robot_config.g(q=q)

            # add in gravity term in task space
            # Jbar = np.dot(M_inv, np.dot(J.T, Mx))
            # g = self.robot_config.g(q=q)
            # self.u_g = g
            # g_task = np.dot(Jbar.T, g)

        if self.null_control:
            # the secondary controller works as a dampener
            Jbar = np.dot(M_inv, np.dot(J.T, Mx))
            u_null = np.dot(M, -self.nkv*dq)
            null_filter = (self.IDENTITY_N_JOINTS - np.dot(J.T, Jbar.T))
            u += np.dot(null_filter, u_null)

        return u
