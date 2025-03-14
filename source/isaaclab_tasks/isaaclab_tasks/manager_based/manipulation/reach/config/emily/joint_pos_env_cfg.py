# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from isaaclab_tasks.manager_based.manipulation.reach.reach_env_cfg import ReachEnvCfg
from isaaclab.assets.articulation import ArticulationCfg

##
# Pre-defined configs
##
from sns_lab.robots.emily import EMILY_CFG, EMILY_INIT_JOINTS


##
# Environment configuration
##


@configclass
class EmilyReachEnvCfg(ReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # switch robot to emily
        self.scene.robot = EMILY_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        
        # override events
        self.events.reset_robot_joints.params["position_range"] = (0.75, 1.25)
        # override rewards
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["ra_flange"]
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = ["ra_flange"]
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = ["ra_flange"]
        # override actions
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=[".*"], scale=1.0, use_default_offset=False
        )
        # override command generator body
        # end-effector is along x-direction
        self.commands.ee_pose.body_name = "ra_flange"
        self.commands.ee_pose.ranges.pitch = (math.pi / 2, math.pi / 2)

@configclass
class EmilyReachEnvCfg_PLAY(EmilyReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False