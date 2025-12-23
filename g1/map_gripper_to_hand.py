import numpy as np

Dex3_Num_Motors = 7

def map_gripper_to_hand(left_gripper_value, right_gripper_value):
        # self.running = True

        q = 0.0
        dq = 0.0
        tau = 0.0
        kp = 1.5
        kd = 0.2

        # initialize dex3-1's left hand cmd msg
        # self.left_msg  = unitree_hg_msg_dds__HandCmd_()
        # for id in Dex3_1_Left_JointIndex:
        #     ris_mode = self._RIS_Mode(id = id, status = 0x01)
        #     motor_mode = ris_mode._mode_to_uint8()
        #     self.left_msg.motor_cmd[id].mode = motor_mode
        #     self.left_msg.motor_cmd[id].q    = q
        #     self.left_msg.motor_cmd[id].dq   = dq
        #     self.left_msg.motor_cmd[id].tau  = tau
        #     self.left_msg.motor_cmd[id].kp   = kp
        #     self.left_msg.motor_cmd[id].kd   = kd

        # # initialize dex3-1's right hand cmd msg
        # self.right_msg = unitree_hg_msg_dds__HandCmd_()
        # for id in Dex3_1_Right_JointIndex:
        #     ris_mode = self._RIS_Mode(id = id, status = 0x01)
        #     motor_mode = ris_mode._mode_to_uint8()
        #     self.right_msg.motor_cmd[id].mode = motor_mode
        #     self.right_msg.motor_cmd[id].q    = q
        #     self.right_msg.motor_cmd[id].dq   = dq
        #     self.right_msg.motor_cmd[id].tau  = tau
        #     self.right_msg.motor_cmd[id].kp   = kp
        #     self.right_msg.motor_cmd[id].kd   = kd

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

        return left_q_target, right_q_target