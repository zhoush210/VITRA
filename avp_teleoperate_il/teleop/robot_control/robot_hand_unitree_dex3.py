

# for dex3-1
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize # dds
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, HandState_                               # idl
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__HandCmd_


import numpy as np
from enum import IntEnum
import time
import os
import sys
import threading
from multiprocessing import Process, shared_memory, Array, Lock

parent2_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parent2_dir)
from teleop.robot_control.hand_retargeting import HandRetargeting, HandType
from teleop.utils.weighted_moving_filter import WeightedMovingFilter

from teleop.image_server.image_client import ImageClient
import cv2

unitree_tip_indices = [4, 9, 14] # [thumb, index, middle] in OpenXR
Dex3_Num_Motors = 7
kTopicDex3LeftCommand = "rt/dex3/left/cmd"
kTopicDex3RightCommand = "rt/dex3/right/cmd"
kTopicDex3LeftState = "rt/dex3/left/state"
kTopicDex3RightState = "rt/dex3/right/state"
class Dex3_1_Left_JointIndex(IntEnum):
    kLeftHandThumb0 = 0
    kLeftHandThumb1 = 1
    kLeftHandThumb2 = 2
    kLeftHandMiddle0 = 3
    kLeftHandMiddle1 = 4
    kLeftHandIndex0 = 5
    kLeftHandIndex1 = 6

class Dex3_1_Right_JointIndex(IntEnum):
    kRightHandThumb0 = 0
    kRightHandThumb1 = 1
    kRightHandThumb2 = 2
    kRightHandIndex0 = 3
    kRightHandIndex1 = 4
    kRightHandMiddle0 = 5
    kRightHandMiddle1 = 6

class Dex3_1_Controller:
    def __init__(self, right_hand_state_array=None, left_hand_state_array=None, dual_hand_action_array=None, fps=100.0, Unit_Test=False):
        self.fps = fps
        self.Unit_Test = Unit_Test

        # 自动设置5秒后打印关节状态
        self._print_state_after = 5

        # 初始化手爪状态和动作数组
        self.left_hand_state_array = left_hand_state_array
        self.right_hand_state_array = right_hand_state_array
        self.left_hand_action_array = dual_hand_action_array  # 左手动作
        self.right_hand_action_array = dual_hand_action_array  # 右手动作

        # 初始化手爪控制器的发布器和订阅器
        self.LeftHandCmb_publisher = ChannelPublisher("rt/dex3/left/cmd", HandCmd_)
        self.LeftHandCmb_publisher.Init()
        self.RightHandCmb_publisher = ChannelPublisher("rt/dex3/right/cmd", HandCmd_)
        self.RightHandCmb_publisher.Init()

        self.LeftHandState_subscriber = ChannelSubscriber("rt/dex3/left/state", HandState_)
        self.LeftHandState_subscriber.Init()
        self.RightHandState_subscriber = ChannelSubscriber("rt/dex3/right/state", HandState_)
        self.RightHandState_subscriber.Init()

        # 启动线程接收手爪状态
        self.subscribe_state_thread = threading.Thread(target=self._subscribe_hand_state)
        self.subscribe_state_thread.daemon = True
        self.subscribe_state_thread.start()

        # 等待手爪状态初始化完成
        while True:
            if any(self.left_hand_state_array) and any(self.right_hand_state_array):
                break
            time.sleep(0.01)
            print("[Dex3_1_Controller] Waiting for DDS data...")

        # 初始化控制消息（只做一次）
        self._initialize_control_messages()

        print("Dex3_1_Controller initialization complete!")

    def _initialize_control_messages(self):
        """初始化控制消息，只调用一次"""
        q = 0.0
        dq = 0.0
        tau = 0.0
        
        kp = 1.5
        kd = 0.2
        
        # kp = 0.0
        # kd = 0.0

        # initialize dex3-1's left hand cmd msg
        self.left_msg  = unitree_hg_msg_dds__HandCmd_()
        for id in Dex3_1_Left_JointIndex:
            ris_mode = self._RIS_Mode(id = id, status = 0x01)
            motor_mode = ris_mode._mode_to_uint8()
            self.left_msg.motor_cmd[id].mode = motor_mode
            self.left_msg.motor_cmd[id].q    = q
            self.left_msg.motor_cmd[id].dq   = dq
            self.left_msg.motor_cmd[id].tau  = tau
            self.left_msg.motor_cmd[id].kp   = kp  # 修改：启用左手的 PD 控制
            self.left_msg.motor_cmd[id].kd   = kd  # 修改：启用左手的 PD 控制

        # initialize dex3-1's right hand cmd msg
        self.right_msg = unitree_hg_msg_dds__HandCmd_()
        for id in Dex3_1_Right_JointIndex:
            ris_mode = self._RIS_Mode(id = id, status = 0x01)
            motor_mode = ris_mode._mode_to_uint8()
            self.right_msg.motor_cmd[id].mode = motor_mode  
            self.right_msg.motor_cmd[id].q    = q
            self.right_msg.motor_cmd[id].dq   = dq
            self.right_msg.motor_cmd[id].tau  = tau
            self.right_msg.motor_cmd[id].kp   = kp
            self.right_msg.motor_cmd[id].kd   = kd

    def _subscribe_hand_state(self):
        """
        订阅手爪状态：读取左右手的电机状态并更新到共享数组
        """
        print_time = None
        has_printed = False
        while True:
            left_hand_msg = self.LeftHandState_subscriber.Read()
            right_hand_msg = self.RightHandState_subscriber.Read()

            # 正常写入共享内存
            if left_hand_msg is not None and right_hand_msg is not None:
                for idx, id in enumerate(Dex3_1_Left_JointIndex):
                    self.left_hand_state_array[idx] = left_hand_msg.motor_state[id].q
                for idx, id in enumerate(Dex3_1_Right_JointIndex):
                    self.right_hand_state_array[idx] = right_hand_msg.motor_state[id].q

            # 记录下发指令5秒后的状态
            if print_time is None and hasattr(self, '_print_state_after'):
                print_time = time.time() + self._print_state_after
            if print_time is not None and not has_printed and time.time() > print_time:
                print("[5s after command] Raw left hand msg:", left_hand_msg)
                print("[5s after command] Raw right hand msg:", right_hand_msg)
                for idx, id in enumerate(Dex3_1_Left_JointIndex):
                    print(f"[5s after command] Left joint {id}: {left_hand_msg.motor_state[id].q}")
                for idx, id in enumerate(Dex3_1_Right_JointIndex):
                    print(f"[5s after command] Right joint {id}: {right_hand_msg.motor_state[id].q}")
                has_printed = True

            time.sleep(0.002)
        


    def ctrl_dual_hand(self, left_q_target, right_q_target):
        """
        设置当前左右手电机的目标关节角度
        """
        # print(f"Ctrl Dual Hand: Left: {left_q_target}, Right: {right_q_target}")
        # 设置左手目标角度
        for idx, id in enumerate(Dex3_1_Left_JointIndex):
            self.left_msg.motor_cmd[id].q = left_q_target[idx]

        # 设置右手目标角度
        for idx, id in enumerate(Dex3_1_Right_JointIndex):
            self.right_msg.motor_cmd[id].q = right_q_target[idx]

        # 发布左手和右手的控制指令
        self.LeftHandCmb_publisher.Write(self.left_msg)
        self.RightHandCmb_publisher.Write(self.right_msg)
    
    def map_gripper_to_hand(self, left_gripper_value, right_gripper_value):
        self.running = True

        q = 0.0
        dq = 0.0
        tau = 0.0
        kp = 1.5
        kd = 0.2

        # initialize dex3-1's left hand cmd msg
        self.left_msg  = unitree_hg_msg_dds__HandCmd_()
        for id in Dex3_1_Left_JointIndex:
            ris_mode = self._RIS_Mode(id = id, status = 0x01)
            motor_mode = ris_mode._mode_to_uint8()
            self.left_msg.motor_cmd[id].mode = motor_mode
            self.left_msg.motor_cmd[id].q    = q
            self.left_msg.motor_cmd[id].dq   = dq
            self.left_msg.motor_cmd[id].tau  = tau
            self.left_msg.motor_cmd[id].kp   = kp
            self.left_msg.motor_cmd[id].kd   = kd

        # initialize dex3-1's right hand cmd msg
        self.right_msg = unitree_hg_msg_dds__HandCmd_()
        for id in Dex3_1_Right_JointIndex:
            ris_mode = self._RIS_Mode(id = id, status = 0x01)
            motor_mode = ris_mode._mode_to_uint8()
            self.right_msg.motor_cmd[id].mode = motor_mode
            self.right_msg.motor_cmd[id].q    = q
            self.right_msg.motor_cmd[id].dq   = dq
            self.right_msg.motor_cmd[id].tau  = tau
            self.right_msg.motor_cmd[id].kp   = kp
            self.right_msg.motor_cmd[id].kd   = kd

        # 7 个自由度
        #left_q_target  = np.full(Dex3_Num_Motors, 0)
        #right_q_target = np.full(Dex3_Num_Motors, 0)
        # 顺序
        """
        # teleop/robot_control/hand_retargeting.py
        self.left_dex3_api_joint_names  = [
            'left_hand_thumb_0_joint',
            'left_hand_thumb_1_joint',
            'left_hand_thumb_2_joint',
            'left_hand_middle_0_joint',
            'left_hand_middle_1_joint',
            'left_hand_index_0_joint',
            'left_hand_index_1_joint' ]
        """
        # 关节角度全是0的情况下，拇指对其另外两指的空隙，90度角打开，
        # 定义 gripper 全打开状态，让拇指微微弯曲，：
            # right_hand_thumb_1_joint: -0.507
            # right_hand_thumb_2_joint: -0.628
            # left_hand_thumb_1_joint: 0.507
            # left_hand_thumb_2_joint: 0.628
        THUMB_1_OPEN = -0.72 #0.507 # 打开时大拇指稍远一点
        THUMB_2_OPEN = 0.0
        # 定义 gripper 全关闭状态, 拇指和两指 指尖在一个平面上:
            # left_hand_thumb_1_joint: 0.888
            # left_hand_thumb_2_joint: 0.628
            # left_hand_middle_0_joint: -0.707
            # left_hand_middle_1_joint: -0.768
            # left_hand_index_0_joint: -0.707
            # left_hand_index_1_joint: -0.768
            # right_hand_thumb_1_joint: -0.888
            # right_hand_thumb_2_joint: -0.628
            # right_hand_middle_0_joint: 0.707
            # right_hand_middle_1_joint: 0.768
            # right_hand_index_0_joint: 0.707
            # right_hand_index_1_joint: 0.768
        THUMB_1_CLOSE = 0.2
        THUMB_2_CLOSE = 0.4
        MIDDLE_0_CLOSE = 0.8
        MIDDLE_1_CLOSE = 1.2
        INDEX_0_CLOSE = 0.8
        INDEX_1_CLOSE = 1.2

        # 构建全打开的初始状态为默认
        left_q_target  = np.full(Dex3_Num_Motors, 0.)
        right_q_target = np.full(Dex3_Num_Motors, 0.)
        left_q_target[1] = THUMB_1_OPEN
        left_q_target[2] = THUMB_2_OPEN
        right_q_target[1] = -THUMB_1_OPEN
        right_q_target[2] = -THUMB_2_OPEN

        # get dual trigger command from XR device
        # with left_gripper_value_in.get_lock():
        #     left_gripper_value  = left_gripper_value_in.value
        # with right_gripper_value_in.get_lock():
        #     right_gripper_value = right_gripper_value_in.value
        # in the following, we map the gripper value [0.0, 1.0] to the hand action
        #logger_mp.info("left right gripper value: %s, %s" % (left_gripper_value, right_gripper_value))
        # Read left and right q_state from shared arrays
        # state_data = np.concatenate((np.array(left_hand_state_array[:]), np.array(right_hand_state_array[:])))

        if left_gripper_value != 0.0 or right_gripper_value != 0.0: # if input data has been initialized.
            # 'left_hand_thumb_1_joint',
            left_q_target[1] = np.interp(left_gripper_value, [0.0, 1.0], [THUMB_1_OPEN, THUMB_1_CLOSE])
            # 'left_hand_thumb_2_joint',
            left_q_target[2] = np.interp(left_gripper_value, [0.0, 1.0], [THUMB_2_OPEN, THUMB_2_CLOSE])
            # 'left_hand_middle_0_joint',
            left_q_target[3] = np.interp(left_gripper_value, [0.0, 1.0], [0.0, -MIDDLE_0_CLOSE])
            left_q_target[4] = np.interp(left_gripper_value, [0.0, 1.0], [0.0, -MIDDLE_1_CLOSE])
            left_q_target[5] = np.interp(left_gripper_value, [0.0, 1.0], [0.0, -INDEX_0_CLOSE])
            left_q_target[6] = np.interp(left_gripper_value, [0.0, 1.0], [0.0, -INDEX_1_CLOSE])

            # 'right_hand_thumb_1_joint',
            right_q_target[1] = np.interp(right_gripper_value, [0.0, 1.0], [-THUMB_1_OPEN, -THUMB_1_CLOSE])
            # 'right_hand_thumb_2_joint',
            right_q_target[2] = np.interp(right_gripper_value, [0.0, 1.0], [-THUMB_2_OPEN, -THUMB_2_CLOSE])
            # 'right_hand_middle_0_joint',
            right_q_target[3] = np.interp(right_gripper_value, [0.0, 1.0], [0.0, MIDDLE_0_CLOSE])
            right_q_target[4] = np.interp(right_gripper_value, [0.0, 1.0], [0.0, MIDDLE_1_CLOSE])
            right_q_target[5] = np.interp(right_gripper_value, [0.0, 1.0], [0.0, INDEX_0_CLOSE])
            right_q_target[6] = np.interp(right_gripper_value, [0.0, 1.0], [0.0, INDEX_1_CLOSE])

        # get dual hand action
        # action_data = np.concatenate((left_q_target, right_q_target))
        # #logger_mp.info("action data: %s" % action_data)
        # if dual_hand_state_array_out and dual_hand_action_array_out:
        #     with dual_hand_data_lock:
        #         dual_hand_state_array_out[:] = state_data
        #         dual_hand_action_array_out[:] = action_data

        self.ctrl_dual_hand(left_q_target, right_q_target)

    class _RIS_Mode:
        def __init__(self, id=0, status=0x01, timeout=0):
            self.motor_mode = 0 
            self.id = id & 0x0F  # 4 bits for id
            self.status = status & 0x07  # 3 bits for status
            self.timeout = timeout & 0x01  # 1 bit for timeout

        def _mode_to_uint8(self):
            self.motor_mode |= (self.id & 0x0F)
            self.motor_mode |= (self.status & 0x07) << 4
            self.motor_mode |= (self.timeout & 0x01) << 7
            return self.motor_mode


if __name__ == "__main__":

    ChannelFactoryInitialize(0)


    tv_img_shape = (720, 1280, 3)
    img_shm = shared_memory.SharedMemory(create=True, size=np.prod(tv_img_shape) * np.uint8().itemsize)
    img_array = np.ndarray(tv_img_shape, dtype=np.uint8, buffer=img_shm.buf)
    client = ImageClient(tv_img_shape=tv_img_shape, tv_img_shm_name=img_shm.name, server_address='192.168.123.164', Unit_Test=False)
    # client = ImageClient(image_show = True, server_address='192.168.123.164', Unit_Test=False) # deployment test
    image_receive_thread = threading.Thread(target = client.receive_process, daemon = True)
    image_receive_thread.daemon = True
    image_receive_thread.start()


    left_state_arr  = Array('d', 7, lock=False)
    right_state_arr = Array('d', 7, lock=False)
    action_arr = Array('d', 14, lock=False)   # 可选，用不到也可以 None
    state_lock = Lock()

    # -------- 3. 创建手爪控制器 --------
    hand_ctrl = Dex3_1_Controller(
        right_hand_state_array  = right_state_arr,
        left_hand_state_array  = left_state_arr,
        dual_hand_action_array = action_arr,
        fps        = 100.0,     # 控制频率
        Unit_Test  = False      # 真实 DDS/真手爪
    )



    lopen_pose  = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    lclose_pose = np.array([0.0, 0.5, 1.0, -1.0, -1.0, -1.0, -1.0])

    ropen_pose  = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    rclose_pose = np.array([0.0, -0.5, -1.0, 1.0, 1.0, 1.0, 1.0])

    while True:
        # 张开
        img = img_array.copy()   # 最新帧
        # 这里可以做OpenCV显示/推理等
        cv2.imshow("img", img)
        cv2.waitKey(1)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
        # hand_ctrl.ctrl_dual_hand(lopen_pose,  ropen_pose)
        # time.sleep(2)
        # hand_ctrl.ctrl_dual_hand(lclose_pose,  rclose_pose)
        # time.sleep(2)
   
        time.sleep(0.02)
