# coding=utf-8
"""
Given a json episode of two arms (collected using xr_teleoperate), visualize in meshcat
"""
import argparse
import json
import meshcat.geometry as mg
import numpy as np
import pinocchio as pin
import time
from pinocchio.robot_wrapper import RobotWrapper
from pinocchio.visualize import MeshcatVisualizer
import os
import sys
import select # For non-blocking input
import tty    # For non-blocking input
import termios # For non-blocking input
import cv2

parser = argparse.ArgumentParser()

parser.add_argument("episode_json")
parser.add_argument("urdf")
parser.add_argument("--image_path", default=None, help="add this to visualize image at the same time")
parser.add_argument("--fps", type=float, default="60", help="the episode is recored in this fps, so we play in this fps")
parser.add_argument("--hand_type", default="dex3")
parser.add_argument("--print_urdf_joints", action="store_true")
parser.add_argument("--bino", action="store_true", help="visualize binocular images")
parser.add_argument("--use_waist", action="store_true", help="visualize waist data")
parser.add_argument("--show_states", action="store_true", help="show states instead of actions")

""" # joint id for reduced g1 with inspire hand
Joint ID 1: left_shoulder_pitch_joint
Joint ID 2: left_shoulder_roll_joint
Joint ID 3: left_shoulder_yaw_joint
Joint ID 4: left_elbow_joint
Joint ID 5: left_wrist_roll_joint
Joint ID 6: left_wrist_pitch_joint
Joint ID 7: left_wrist_yaw_joint
Joint ID 8: L_index_proximal_joint
Joint ID 9: L_middle_proximal_joint
Joint ID 10: L_pinky_proximal_joint
Joint ID 11: L_ring_proximal_joint
Joint ID 12: L_thumb_proximal_yaw_joint
Joint ID 13: L_thumb_proximal_pitch_joint
Joint ID 14: right_shoulder_pitch_joint
Joint ID 15: right_shoulder_roll_joint
Joint ID 16: right_shoulder_yaw_joint
Joint ID 17: right_elbow_joint
Joint ID 18: right_wrist_roll_joint
Joint ID 19: right_wrist_pitch_joint
Joint ID 20: right_wrist_yaw_joint
Joint ID 21: R_index_proximal_joint
Joint ID 22: R_middle_proximal_joint
Joint ID 23: R_pinky_proximal_joint
Joint ID 24: R_ring_proximal_joint
Joint ID 25: R_thumb_proximal_yaw_joint
Joint ID 26: R_thumb_proximal_pitch_joint
"""

""" # joint id for reduced g1 with dex3 hand
# to print it, use this command:
    (g1) junweil@office-precognition:~/projects/humanoid_teleop$ python g1_realrobot/visualize_arm_episodes.py ~/projects/test2/xr_teleoperate/teleop//utils/data/episode_0014/data.json assets/g1/g1_body29_hand14.urdf --hand_type dex3 --fps 60 --print

Joint ID 1: left_shoulder_pitch_joint
Joint ID 2: left_shoulder_roll_joint
Joint ID 3: left_shoulder_yaw_joint
Joint ID 4: left_elbow_joint
Joint ID 5: left_wrist_roll_joint
Joint ID 6: left_wrist_pitch_joint
Joint ID 7: left_wrist_yaw_joint
Joint ID 8: left_hand_index_0_joint
Joint ID 9: left_hand_index_1_joint
Joint ID 10: left_hand_middle_0_joint
Joint ID 11: left_hand_middle_1_joint
Joint ID 12: left_hand_thumb_0_joint
Joint ID 13: left_hand_thumb_1_joint
Joint ID 14: left_hand_thumb_2_joint
Joint ID 15: right_shoulder_pitch_joint
Joint ID 16: right_shoulder_roll_joint
Joint ID 17: right_shoulder_yaw_joint
Joint ID 18: right_elbow_joint
Joint ID 19: right_wrist_roll_joint
Joint ID 20: right_wrist_pitch_joint
Joint ID 21: right_wrist_yaw_joint
Joint ID 22: right_hand_index_0_joint
Joint ID 23: right_hand_index_1_joint
Joint ID 24: right_hand_middle_0_joint
Joint ID 25: right_hand_middle_1_joint
Joint ID 26: right_hand_thumb_0_joint
Joint ID 27: right_hand_thumb_1_joint
Joint ID 28: right_hand_thumb_2_joint

"""

class G1_29_Vis_Episode:
    def __init__(self, urdf, hand_type="inspire1", print_urdf_joints=False):

        np.set_printoptions(precision=5, suppress=True, linewidth=200)

        self.robot = pin.RobotWrapper.BuildFromURDF(urdf, os.path.dirname(urdf))

        # 五指手，三指手
        assert hand_type in ["inspire1", "dex3"]

        if hand_type == "inspire1":
            self.mixed_jointsToLockIDs = [
                # 固定下半身
                "left_hip_pitch_joint" ,
                "left_hip_roll_joint" ,
                "left_hip_yaw_joint" ,
                "left_knee_joint" ,
                "left_ankle_pitch_joint" ,
                "left_ankle_roll_joint" ,
                "right_hip_pitch_joint" ,
                "right_hip_roll_joint" ,
                "right_hip_yaw_joint" ,
                "right_knee_joint" ,
                "right_ankle_pitch_joint" ,
                "right_ankle_roll_joint" ,

                # 腰部不要锁，可能有数值
                #"waist_yaw_joint" ,
                #"waist_roll_joint" ,
                #"waist_pitch_joint",


                # 单手URDF里，12个自由度，4个手指每个2个所以8个，剩4个自由度在拇指
                # 实机单手只有6自由度，每个手指一个，拇指2个

                # 这六个是主动关节， 我们锁定其他的被动关节
                # 遥操作的时候也只有6个主动关节的数据
                #'R_thumb_proximal_yaw_joint',
                #'R_thumb_proximal_pitch_joint',
                #'R_index_proximal_joint',
                #'R_middle_proximal_joint',
                #'R_ring_proximal_joint',
                #'R_pinky_proximal_joint'

                # 左手关节
                #"L_pinky_proximal_joint",
                "L_pinky_intermediate_joint",
                #"L_ring_proximal_joint",
                "L_ring_intermediate_joint",
                "L_thumb_intermediate_joint",
                #"L_thumb_proximal_yaw_joint",
                #"L_thumb_proximal_pitch_joint",
                "L_thumb_distal_joint",
                #"L_middle_proximal_joint",
                "L_middle_intermediate_joint",
                #"L_index_proximal_joint",
                "L_index_intermediate_joint",

                # 右手关节（已更新）
                #"R_pinky_proximal_joint",
                "R_pinky_intermediate_joint",
                #"R_ring_proximal_joint",
                "R_ring_intermediate_joint",
                "R_thumb_intermediate_joint",
                #"R_thumb_proximal_yaw_joint",
                #"R_thumb_proximal_pitch_joint",
                "R_thumb_distal_joint",
                #"R_index_proximal_joint",
                "R_index_intermediate_joint",
                #"R_middle_proximal_joint",
                "R_middle_intermediate_joint"
            ]
        elif hand_type == "dex3":

            self.mixed_jointsToLockIDs = [
                # 固定下半身
                "left_hip_pitch_joint" ,
                "left_hip_roll_joint" ,
                "left_hip_yaw_joint" ,
                "left_knee_joint" ,
                "left_ankle_pitch_joint" ,
                "left_ankle_roll_joint" ,
                "right_hip_pitch_joint" ,
                "right_hip_roll_joint" ,
                "right_hip_yaw_joint" ,
                "right_knee_joint" ,
                "right_ankle_pitch_joint" ,
                "right_ankle_roll_joint" ,
                #"waist_yaw_joint" ,
                #"waist_roll_joint" ,
                #"waist_pitch_joint",

                # 用的宇树三指手的URDF，每个7自由度，都不用锁

            ]


        # https://docs.ros.org/en/kinetic/api/pinocchio/html/classpinocchio_1_1robot__wrapper_1_1RobotWrapper.html#aef341b27b4709b03c93d66c8c196bc0f
        # the above joint will be locked, at 0.0
        self.reduced_robot = self.robot.buildReducedRobot(
            list_of_joints_to_lock=self.mixed_jointsToLockIDs,
            reference_configuration=np.array([0.0] * self.robot.model.nq),
        )

        #debugging printouts
        if print_urdf_joints:
            print("reduced_robot.model.nframes")
            for i in range(self.reduced_robot.model.nframes):
                frame = self.reduced_robot.model.frames[i]
                frame_id = self.reduced_robot.model.getFrameId(frame.name)
                print(f"Frame ID: {frame_id}, Name: {frame.name}")

            #assert len(self.reduced_robot.model.frames) == len(self.reduced_robot.data.oMf), \
            #    f"Mismatch: {len(self.reduced_robot.model.frames)} frames vs. {len(self.reduced_robot.data.oMf)} transformations"

            # Print all joints in the original robot model
            print("All Joints in Original Robot:")
            for idx, joint in enumerate(self.robot.model.names):
                print(f"Joint ID {idx}: {joint}")

            # Print joints in the reduced robot model
            print("\nJoints in Reduced Robot:")
            for idx, joint in enumerate(self.reduced_robot.model.names):
                print(f"Joint ID {idx}: {joint}")

            print("reduced_robot.model.nq:%s" % self.reduced_robot.model.nq)
            sys.exit()


        self.init_data = np.zeros(self.reduced_robot.model.nq)

        self.current_q = np.zeros(self.reduced_robot.model.nq) # used to save the current q


        # Initialize the Meshcat visualizer for visualization
        self.vis = MeshcatVisualizer(
            self.reduced_robot.model, self.reduced_robot.collision_model, self.reduced_robot.visual_model)
        self.vis.initViewer(open=True)
        self.vis.loadViewerModel("pinocchio")

        self.vis.display(pin.neutral(self.reduced_robot.model))

class G1_29_Vis_WholeBody:
    def __init__(self, urdf, hand_type="inspire1", print_urdf_joints=False):

        np.set_printoptions(precision=5, suppress=True, linewidth=200)

        self.robot = pin.RobotWrapper.BuildFromURDF(urdf, os.path.dirname(urdf))

        # 五指手，三指手
        assert hand_type in ["inspire1", "dex3"]

        if hand_type == "inspire1":
            self.mixed_jointsToLockIDs = [

                # 单手URDF里，12个自由度，4个手指每个2个所以8个，剩4个自由度在拇指
                # 实机单手只有6自由度，每个手指一个，拇指2个

                # 这六个是主动关节， 我们锁定其他的被动关节
                # 遥操作的时候也只有6个主动关节的数据
                #'R_thumb_proximal_yaw_joint',
                #'R_thumb_proximal_pitch_joint',
                #'R_index_proximal_joint',
                #'R_middle_proximal_joint',
                #'R_ring_proximal_joint',
                #'R_pinky_proximal_joint'

                # 左手关节
                #"L_pinky_proximal_joint",
                "L_pinky_intermediate_joint",
                #"L_ring_proximal_joint",
                "L_ring_intermediate_joint",
                "L_thumb_intermediate_joint",
                #"L_thumb_proximal_yaw_joint",
                #"L_thumb_proximal_pitch_joint",
                "L_thumb_distal_joint",
                #"L_middle_proximal_joint",
                "L_middle_intermediate_joint",
                #"L_index_proximal_joint",
                "L_index_intermediate_joint",

                # 右手关节（已更新）
                #"R_pinky_proximal_joint",
                "R_pinky_intermediate_joint",
                #"R_ring_proximal_joint",
                "R_ring_intermediate_joint",
                "R_thumb_intermediate_joint",
                #"R_thumb_proximal_yaw_joint",
                #"R_thumb_proximal_pitch_joint",
                "R_thumb_distal_joint",
                #"R_index_proximal_joint",
                "R_index_intermediate_joint",
                #"R_middle_proximal_joint",
                "R_middle_intermediate_joint"
            ]
        elif hand_type == "dex3":

            self.mixed_jointsToLockIDs = [

                # 用的宇树三指手的URDF，每个7自由度，都不用锁

            ]

        # https://docs.ros.org/en/kinetic/api/pinocchio/html/classpinocchio_1_1robot__wrapper_1_1RobotWrapper.html#aef341b27b4709b03c93d66c8c196bc0f
        # the above joint will be locked, at 0.0
        self.reduced_robot = self.robot.buildReducedRobot(
            list_of_joints_to_lock=self.mixed_jointsToLockIDs,
            reference_configuration=np.array([0.0] * self.robot.model.nq),
        )

        #debugging printouts
        if print_urdf_joints:
            print("reduced_robot.model.nframes")
            for i in range(self.reduced_robot.model.nframes):
                frame = self.reduced_robot.model.frames[i]
                frame_id = self.reduced_robot.model.getFrameId(frame.name)
                print(f"Frame ID: {frame_id}, Name: {frame.name}")

            #assert len(self.reduced_robot.model.frames) == len(self.reduced_robot.data.oMf), \
            #    f"Mismatch: {len(self.reduced_robot.model.frames)} frames vs. {len(self.reduced_robot.data.oMf)} transformations"

            # Print all joints in the original robot model
            print("All Joints in Original Robot:")
            for idx, joint in enumerate(self.robot.model.names):
                print(f"Joint ID {idx}: {joint}")

            # Print joints in the reduced robot model
            print("\nJoints in Reduced Robot:")
            for idx, joint in enumerate(self.reduced_robot.model.names):
                print(f"Joint ID {idx}: {joint}")

            print("reduced_robot.model.nq:%s" % self.reduced_robot.model.nq)
            sys.exit()

        self.init_data = np.zeros(self.reduced_robot.model.nq)

        self.current_q = np.zeros(self.reduced_robot.model.nq) # used to save the current q

        # Initialize the Meshcat visualizer for visualization
        self.vis = MeshcatVisualizer(
            self.reduced_robot.model, self.reduced_robot.collision_model, self.reduced_robot.visual_model)
        self.vis.initViewer(open=True)
        self.vis.loadViewerModel("pinocchio")

        self.vis.display(pin.neutral(self.reduced_robot.model))
"""
for dex3 whole  body
Joints in Reduced Robot:
Joint ID 0: universe
Joint ID 1: left_hip_pitch_joint
Joint ID 2: left_hip_roll_joint
Joint ID 3: left_hip_yaw_joint
Joint ID 4: left_knee_joint
Joint ID 5: left_ankle_pitch_joint
Joint ID 6: left_ankle_roll_joint
Joint ID 7: right_hip_pitch_joint
Joint ID 8: right_hip_roll_joint
Joint ID 9: right_hip_yaw_joint
Joint ID 10: right_knee_joint
Joint ID 11: right_ankle_pitch_joint
Joint ID 12: right_ankle_roll_joint
Joint ID 13: waist_yaw_joint
Joint ID 14: waist_roll_joint
Joint ID 15: waist_pitch_joint
Joint ID 16: left_shoulder_pitch_joint
Joint ID 17: left_shoulder_roll_joint
Joint ID 18: left_shoulder_yaw_joint
Joint ID 19: left_elbow_joint
Joint ID 20: left_wrist_roll_joint
Joint ID 21: left_wrist_pitch_joint
Joint ID 22: left_wrist_yaw_joint
Joint ID 23: left_hand_index_0_joint
Joint ID 24: left_hand_index_1_joint
Joint ID 25: left_hand_middle_0_joint
Joint ID 26: left_hand_middle_1_joint
Joint ID 27: left_hand_thumb_0_joint
Joint ID 28: left_hand_thumb_1_joint
Joint ID 29: left_hand_thumb_2_joint
Joint ID 30: right_shoulder_pitch_joint
Joint ID 31: right_shoulder_roll_joint
Joint ID 32: right_shoulder_yaw_joint
Joint ID 33: right_elbow_joint
Joint ID 34: right_wrist_roll_joint
Joint ID 35: right_wrist_pitch_joint
Joint ID 36: right_wrist_yaw_joint
Joint ID 37: right_hand_index_0_joint
Joint ID 38: right_hand_index_1_joint
Joint ID 39: right_hand_middle_0_joint
Joint ID 40: right_hand_middle_1_joint
Joint ID 41: right_hand_thumb_0_joint
Joint ID 42: right_hand_thumb_1_joint
Joint ID 43: right_hand_thumb_2_joint
reduced_robot.model.nq:43

for inspire1 whole body
Joints in Reduced Robot:
Joint ID 0: universe
Joint ID 1: left_hip_pitch_joint
Joint ID 2: left_hip_roll_joint
Joint ID 3: left_hip_yaw_joint
Joint ID 4: left_knee_joint
Joint ID 5: left_ankle_pitch_joint
Joint ID 6: left_ankle_roll_joint
Joint ID 7: right_hip_pitch_joint
Joint ID 8: right_hip_roll_joint
Joint ID 9: right_hip_yaw_joint
Joint ID 10: right_knee_joint
Joint ID 11: right_ankle_pitch_joint
Joint ID 12: right_ankle_roll_joint
Joint ID 13: waist_yaw_joint
Joint ID 14: waist_roll_joint
Joint ID 15: waist_pitch_joint
Joint ID 16: left_shoulder_pitch_joint
Joint ID 17: left_shoulder_roll_joint
Joint ID 18: left_shoulder_yaw_joint
Joint ID 19: left_elbow_joint
Joint ID 20: left_wrist_roll_joint
Joint ID 21: left_wrist_pitch_joint
Joint ID 22: left_wrist_yaw_joint
Joint ID 23: L_index_proximal_joint
Joint ID 24: L_middle_proximal_joint
Joint ID 25: L_pinky_proximal_joint
Joint ID 26: L_ring_proximal_joint
Joint ID 27: L_thumb_proximal_yaw_joint
Joint ID 28: L_thumb_proximal_pitch_joint
Joint ID 29: right_shoulder_pitch_joint
Joint ID 30: right_shoulder_roll_joint
Joint ID 31: right_shoulder_yaw_joint
Joint ID 32: right_elbow_joint
Joint ID 33: right_wrist_roll_joint
Joint ID 34: right_wrist_pitch_joint
Joint ID 35: right_wrist_yaw_joint
Joint ID 36: R_index_proximal_joint
Joint ID 37: R_middle_proximal_joint
Joint ID 38: R_pinky_proximal_joint
Joint ID 39: R_ring_proximal_joint
Joint ID 40: R_thumb_proximal_yaw_joint
Joint ID 41: R_thumb_proximal_pitch_joint
reduced_robot.model.nq:41

"""

# Global variable to store old terminal settings
old_terminal_settings = None

# mapping of saved api joints to URDF joints
left_inspire_api_joint_names = [
    'L_pinky_proximal_joint',
    'L_ring_proximal_joint',
    'L_middle_proximal_joint',
    'L_index_proximal_joint',
    'L_thumb_proximal_pitch_joint',
    'L_thumb_proximal_yaw_joint' ]

left_inspire_urdf_joint_names = [
    'L_index_proximal_joint',
    'L_middle_proximal_joint',
    'L_pinky_proximal_joint',
    'L_ring_proximal_joint',
    'L_thumb_proximal_yaw_joint',
    'L_thumb_proximal_pitch_joint',
]
left_inspire_api_to_urdf_index = [
    left_inspire_api_joint_names.index(name)
    for name in left_inspire_urdf_joint_names]

right_inspire_api_joint_names = [
    'R_pinky_proximal_joint',
    'R_ring_proximal_joint',
    'R_middle_proximal_joint',
    'R_index_proximal_joint',
    'R_thumb_proximal_pitch_joint',
    'R_thumb_proximal_yaw_joint' ]

right_inspire_urdf_joint_names = [
    'R_index_proximal_joint',
    'R_middle_proximal_joint',
    'R_pinky_proximal_joint',
    'R_ring_proximal_joint',
    'R_thumb_proximal_yaw_joint',
    'R_thumb_proximal_pitch_joint',
]
right_inspire_api_to_urdf_index = [
    right_inspire_api_joint_names.index(name)
    for name in right_inspire_urdf_joint_names]

left_dex3_api_joint_names = [
    'left_hand_thumb_0_joint',
    'left_hand_thumb_1_joint',
    'left_hand_thumb_2_joint',
    'left_hand_middle_0_joint',
    'left_hand_middle_1_joint',
    'left_hand_index_0_joint',
    'left_hand_index_1_joint' ]

left_dex3_urdf_joint_names = [
    'left_hand_index_0_joint',
    'left_hand_index_1_joint',
    'left_hand_middle_0_joint',
    'left_hand_middle_1_joint',
    'left_hand_thumb_0_joint',
    'left_hand_thumb_1_joint',
    'left_hand_thumb_2_joint' ]
left_dex3_api_to_urdf_index = [
    left_dex3_api_joint_names.index(name)
    for name in left_dex3_urdf_joint_names]

right_dex3_api_joint_names = [
    'right_hand_thumb_0_joint',
    'right_hand_thumb_1_joint',
    'right_hand_thumb_2_joint',
    'right_hand_middle_0_joint',
    'right_hand_middle_1_joint',
    'right_hand_index_0_joint',
    'right_hand_index_1_joint' ]

right_dex3_urdf_joint_names = [
    'right_hand_index_0_joint',
    'right_hand_index_1_joint',
    'right_hand_middle_0_joint',
    'right_hand_middle_1_joint',
    'right_hand_thumb_0_joint',
    'right_hand_thumb_1_joint',
    'right_hand_thumb_2_joint' ]
right_dex3_api_to_urdf_index = [
    right_dex3_api_joint_names.index(name)
    for name in right_dex3_urdf_joint_names]




def set_terminal_cbreak():
    """Sets the terminal to cbreak mode and saves old settings."""
    global old_terminal_settings
    fd = sys.stdin.fileno()
    old_terminal_settings = termios.tcgetattr(fd)
    tty.setcbreak(fd)

def restore_terminal_settings():
    """Restores the terminal to its original settings."""
    global old_terminal_settings
    if old_terminal_settings is not None:
        fd = sys.stdin.fileno()
        termios.tcsetattr(fd, termios.TCSADRAIN, old_terminal_settings)
        old_terminal_settings = None # Clear after restoring

def get_char_nonblocking():
    """Reads a single character from stdin without blocking, for Unix-like systems."""
    fd = sys.stdin.fileno()
    # Check if data is available for reading
    if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
        try:
            char = sys.stdin.read(1)
            return char
        except IOError:
            # This can happen if stdin is closed or other issues
            return None
    return None

def show_current_q(vis_model, left_arm_pos, left_ee_pos, right_arm_pos, right_ee_pos):
    left_ee_pos = left_ee_pos[left_dex3_api_to_urdf_index]
    right_ee_pos = right_ee_pos[right_dex3_api_to_urdf_index]
    waist_q = np.zeros((3, ), dtype=np.float32)
    
    target_q = np.zeros((31, ), dtype=np.float32)
    target_q[:3] = waist_q
    target_q[3:10] = left_arm_pos
    target_q[10:17] = left_ee_pos
    target_q[17:24] = right_arm_pos
    target_q[24:] = right_ee_pos

    vis_model.vis.display(target_q)


# xr_teleoperate/teleop/robot_control/robot_hand_inspired.py has the q_target normalized
def denorm_inspire(normed_ee_pos):
    Inspire_Num_Motors = 6
    for idx in range(Inspire_Num_Motors):
        if idx <= 3:
            normed_ee_pos[idx]  = denormalize(normed_ee_pos[idx], 0.0, 1.7)
        elif idx == 4:
            normed_ee_pos[idx]  = denormalize(normed_ee_pos[idx], 0.0, 0.5)
        elif idx == 5:
            normed_ee_pos[idx]  = denormalize(normed_ee_pos[idx], -0.1, 1.3)


def denormalize(normalized_val, min_val, max_val):
    return max_val - (normalized_val * (max_val - min_val))

def resize_image_to_width(image, target_width):
    """
    Resizes an image to a target width while maintaining the aspect ratio.

    Args:
        image (np.ndarray): The input image as a NumPy array.
        target_width (int): The desired width of the resized image.

    Returns:
        np.ndarray: The resized image.
    """
    # Get the original dimensions of the image
    original_height, original_width = image.shape[:2]

    # If the image is already the target width, no need to resize
    if original_width == target_width:
        return image

    # Calculate the aspect ratio
    aspect_ratio = original_height / original_width

    # Calculate the new height based on the target width and aspect ratio
    target_height = int(target_width * aspect_ratio)

    # New dimensions tuple
    new_dimensions = (target_width, target_height)

    # Resize the image using the calculated dimensions
    # cv2.INTER_AREA is generally recommended for shrinking images
    resized_image = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_AREA)

    return resized_image


if __name__ == "__main__":
    args = parser.parse_args()

    vis_model = G1_29_Vis_Episode(urdf=args.urdf, hand_type=args.hand_type, print_urdf_joints=args.print_urdf_joints)

    episode = json.load(open(args.episode_json))
    num_data_step = len(episode["data"])
    print("total %d data steps, it should be %.2f seconds long" % (num_data_step, num_data_step/args.fps))

    current_step = 0
    paused = False

    # Keep track of key press states to avoid multiple triggers on hold
    s_pressed_handled = False
    plus_pressed_handled = False
    minus_pressed_handled = False

    # Set terminal to cbreak mode at the start
    set_terminal_cbreak()

    show_image = False
    if args.image_path is not None:
        show_image = True
        # episode_0001/colors
        # 000825_color_0.jpg # images as many as steps
        # if binocular,
        # there are also 000825_color_1.jpg
    try:
        while current_step < num_data_step:
            start_time = time.time()
            image_file_name_info = "" # To store image name for printing
            if show_image:
                image = None
                if args.bino:
                    # Load and display binocular images
                    image_file_name_left = "%06d_color_0.jpg" % current_step
                    image_file_name_right = "%06d_color_1.jpg" % current_step
                    image_file_left = os.path.join(args.image_path, image_file_name_left)
                    image_file_right = os.path.join(args.image_path, image_file_name_right)

                    if os.path.exists(image_file_left) and os.path.exists(image_file_right):
                        image_left = cv2.imread(image_file_left)
                        image_right = cv2.imread(image_file_right)
                        if image_left is not None and image_right is not None:
                            image = cv2.hconcat([image_left, image_right])
                            image_file_name_info = f"{image_file_name_left}, {image_file_name_right}"
                else:
                    # Load and display single image
                    image_file_name = "%06d_color_0.jpg" % current_step
                    image_file = os.path.join(args.image_path, image_file_name)
                    if os.path.exists(image_file):
                        image = cv2.imread(image_file)
                        image_file_name_info = image_file_name

                if image is not None:
                    # if the data saved the delay times as well
                    if "delay" in episode["data"][current_step]:
                        delay_in_seconds = float(episode["data"][current_step]["delay"])
                        # Add delay text to the resized image
                        delay_text = f"Delay: {delay_in_seconds * 1000:.2f} ms"
                        cv2.putText(
                            image,
                            delay_text,
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (255, 255, 255),
                            2,
                            cv2.LINE_AA
                        )
                    # show the trigger value if any
                    if "left_trigger" in episode["data"][current_step]["actions"]:
                        left_trigger_value = float(episode["data"][current_step]["actions"]["left_trigger"])
                        right_trigger_value = float(episode["data"][current_step]["actions"]["right_trigger"])
                        trigger_text = f"Trigger value, l: {left_trigger_value:.2f}, r: {right_trigger_value:.2f}"
                        cv2.putText(
                            image,
                            trigger_text,
                            (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (255, 0, 255),
                            2,
                            cv2.LINE_AA
                        )
                    # bino image might be too wide, resize
                    if args.bino:
                        image = resize_image_to_width(image, 1920)

                    cv2.imshow("Episode Image", image)
                    cv2.waitKey(1) # Refresh image window

            # char = get_char_nonblocking()
            # if char is not None:
            #     # print(f"Key pressed: '{char}'") # For debugging input
            #     if char == 's':
            #         if not s_pressed_handled:
            #             paused = not paused
            #             print(f"\n{'Paused' if paused else 'Resumed'} replay.")
            #             s_pressed_handled = True
            #     elif char == ',':
            #         if paused and not plus_pressed_handled:
            #             current_step = min(num_data_step - 1, current_step + 10)
            #             print(f"\nStepped forward to step {current_step}")

            #             if args.show_states:
            #                 step_data = episode["data"][current_step]["states"]
            #             else:
            #                 step_data = episode["data"][current_step]["actions"]
            #             show_current_q(
            #                 vis_model, step_data,
            #                 hand_type=args.hand_type, visualize_waist=args.use_waist)

            #             plus_pressed_handled = True
            #     elif char == '.':
            #         if paused and not minus_pressed_handled:
            #             current_step = max(0, current_step - 10)
            #             print(f"\nStepped back to step {current_step}")

            #             if args.show_states:
            #                 step_data = episode["data"][current_step]["states"]
            #             else:
            #                 step_data = episode["data"][current_step]["actions"]
            #             show_current_q(
            #                 vis_model, step_data,
            #                 hand_type=args.hand_type, visualize_waist=args.use_waist)

            #             minus_pressed_handled = True
            #     # Handle Ctrl+C (ASCII 3) or Ctrl+D (ASCII 4) to exit cleanly
            #     elif ord(char) == 3 or ord(char) == 4:
            #         print("\nExiting replay.")
            #         break # Exit the loop

            # # Reset handled flags if keys are released
            # if char != 's':
            #     s_pressed_handled = False
            # if char != '+':
            #     plus_pressed_handled = False
            # if char != '-':
            #     minus_pressed_handled = False


            if not paused:
                current_episode_time = current_step / args.fps
                # Use sys.stdout.write for finer control and avoid print's buffering issues
                if show_image:
                    sys.stdout.write(
                        f"\rTime: {current_episode_time:.2f}s | Step ID: {current_step}/{num_data_step-1} | Image {image_file_name_info}")
                else:
                    sys.stdout.write(
                        f"\rTime: {current_episode_time:.2f}s | Step ID: {current_step}/{num_data_step-1}")
                sys.stdout.flush() # Ensure it's written immediately

                if args.show_states:
                    step_data = episode["data"][current_step]["states"]
                else:
                    import pdb; pdb.set_trace()
                    step_data = episode["data"][current_step]["actions"]
                show_current_q(
                    vis_model, step_data,
                    hand_type=args.hand_type, visualize_waist=args.use_waist)
                current_step += 1

            # Ensure consistent frame rate
            current_time = time.time()
            time_elapsed = current_time - start_time
            sleep_time = max(0, (1 / args.fps) - time_elapsed)
            time.sleep(sleep_time)

    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        # Always restore terminal settings before exiting
        restore_terminal_settings()
        if show_image:
            cv2.destroyAllWindows() # Close all OpenCV windows
        print("\nReplay finished.")

