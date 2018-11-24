"""
Running operational space control using VREP. The controller will
move the end-effector to the target object's orientation.
"""
import numpy as np
import pygame

from abr_control.arms import ur5 as arm
# from abr_control.arms import jaco2 as arm
from abr_control.controllers import OSC
from abr_control.interfaces import VREP
from abr_control.utils import transformations


# initialize our robot config
robot_config = arm.Config(use_cython=True)
# create opreational space controller
ctrlr = OSC(robot_config, kp=500)

# create our interface
interface = VREP(robot_config, dt=.005)
interface.connect()

# set up lists for tracking data
ee_angles_track = []
target_angles_track = []

# control (alpha, beta, gamma) out of [x, y, z, alpha, beta, gamma]
ctrlr_dof = [False, False, False, True, True, True]


try:
    count = 0
    print('\nSimulation starting...\n')
    while 1:
        # get arm feedback
        feedback = interface.get_feedback()
        hand_xyz = robot_config.Tx('EE', feedback['q'])

        target = np.hstack([
            interface.get_xyz('target'),
            interface.get_orientation('target')])

        vrep_angles = interface.get_orientation('UR5_link6')
        vrep_R = transformations.euler_matrix(
            vrep_angles[0], vrep_angles[1], vrep_angles[2], 'rxyz')
        rc_matrix = robot_config.R('EE', feedback['q'])
        rc_angles = transformations.euler_from_matrix(rc_matrix, axes='rxyz')
        interface.set_orientation('object', rc_angles)

        print()
        print()
        print('vrep matrix: \n', vrep_R)
        print('rc matrix: \n', rc_matrix)
        print()
        print('vrep angles: ', vrep_angles)
        print('rc angles: ', list(rc_angles))

        # interface.set_xyz('object', robot_config.Tx(name, q=feedback['q']))
        # interface.set_orientation('object', vrep_angles)
        #
        # print('rc angles: ', [float('%.3f' % val) for val in np.array(rc_angles) * 180 / np.pi])
        # print('VREP angles: ', [float('%.3f' % val) for val in np.array(vrep_angles) * 180 / np.pi])
        # # print()
        # print('rc matrix: \n')
        # for row in rc_matrix:
        #     print([float('%.3f' % val) for val in row])
        # print('VREP matrix : \n')
        # vrep_matrix = transformations.euler_matrix(
        #     vrep_angles[0], vrep_angles[1], vrep_angles[2], axes='rxyz')[:3, :3]
        # for row in vrep_matrix:
        #     print([float('%.3f' % val) for val in row])

        u = ctrlr.generate(
            q=feedback['q'],
            dq=feedback['dq'],
            target=target,
            ctrlr_dof=ctrlr_dof,
            )

        # apply the control signal, step the sim forward
        interface.send_forces(u)

        # track data
        ee_angles_track.append(transformations.euler_from_matrix(
            robot_config.R('EE', feedback['q'])))
        target_angles_track.append(interface.get_orientation('target'))
        count += 1

finally:
    # stop and reset the simulation
    interface.disconnect()

    print('Simulation terminated...')

    ee_angles_track = np.array(ee_angles_track)
    target_angles_track = np.array(target_angles_track)

    if ee_angles_track.shape[0] > 0:
        # plot distance from target and 3D trajectory
        import matplotlib.pyplot as plt
        from abr_control.utils.plotting import plot_3D

        plt.figure()
        plt.plot(ee_angles_track)
        plt.gca().set_prop_cycle(None)
        plt.plot(target_angles_track, '--')
        plt.ylabel('3D orientation (rad)')
        plt.xlabel('Time (s)')
        plt.tight_layout()
        plt.show()
