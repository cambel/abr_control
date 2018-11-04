"""
Running the threelink arm with the pygame display. The arm will
move the end-effector to the target, which can be moved by
clicking on the background.
"""
import numpy as np
import pygame

from abr_control.arms import threejoint as arm
# from abr_control.arms import twojoint as arm
from abr_control.interfaces import PyGame
from abr_control.utils import transformations
from abr_control.controllers import OSC


# initialize our robot config
robot_config = arm.Config(use_cython=True)
ctrlr = OSC(robot_config, kp=50, vmax=None)
# create our arm simulation
arm_sim = arm.ArmSim(robot_config)


def on_click(self, mouse_x, mouse_y):
    self.target[0] = self.mouse_x
    self.target[1] = self.mouse_y


def on_keypress(self, key):
    if key == pygame.K_LEFT:
        self.theta += np.pi / 10
    if key == pygame.K_RIGHT:
        self.theta -= np.pi / 10
    print('theta: ', self.theta)

    R_theta = np.array([
        [np.cos(interface.theta), -np.sin(interface.theta), 0],
        [np.sin(interface.theta), np.cos(interface.theta), 0],
        [0, 0, 1]])
    R_target = np.dot(R_theta, R)
    self.target_angles = transformations.euler_from_matrix(R_target, axes='sxyz')


# create our interface
interface = PyGame(robot_config, arm_sim, dt=.001,
                   on_click=on_click, on_keypress=on_keypress)
interface.connect()
feedback = interface.get_feedback()
# set target position
target_xyz = robot_config.Tx('EE', feedback['q'])
interface.set_target(target_xyz)
# set target orientation
interface.theta = - 3 * np.pi / 4
R = robot_config.R('EE', feedback['q'])
interface.on_keypress(interface, None)

# set up lists for tracking data
ee_path = []
target_path = []

try:
    # run ctrl.generate once to load all functions
    zeros = np.zeros(robot_config.N_JOINTS)
    robot_config.R('EE', q=zeros)

    print('\nSimulation starting...\n')
    print('\nPress left or right arrow to change target orientation angle.\n')

    count = 0
    while 1:
        # get arm feedback
        feedback = interface.get_feedback()
        hand_xyz = robot_config.Tx('EE', feedback['q'])

        u = ctrlr.generate(
            q=feedback['q'],
            dq=feedback['dq'],
            target_pos=np.hstack([target_xyz[0], interface.target_angles[1:]]),
            target_vel=0,
            ctrlr_dof=[True, False, False, False, True, True],
            )

        new_target = interface.get_mousexy()
        if new_target is not None:
            target_xyz[0:2] = new_target
        interface.set_target(target_xyz)

        # apply the control signal, step the sim forward
        interface.send_forces(
            u, update_display=True if count % 20 == 0 else False)

        # track data
        ee_path.append(np.copy(hand_xyz))
        # target_path.append(np.copy(target_xyz))
        count += 1

finally:
    # stop and reset the simulation
    interface.disconnect()

    print('Simulation terminated...')
