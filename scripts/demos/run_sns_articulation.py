# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates different single-arm manipulators.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/demos/arms.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates different single-arm manipulators.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import numpy as np
import torch

import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation

#import omni
#ext_manager = omni.kit.app.get_app().get_extension_manager()
#ext_manager.set_extension_enabled_immediate("omni.isaac.ros2_bridge", True)
##
# Pre-defined configs
##
# isort: off

# import Shennongshi lab robots
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from sns_lab.robots.emily import EMILY_CFG
from isaaclab_assets import UR10_CFG

# isort: on


def design_scene() -> tuple[dict, list[list[float]]]:
    """Designs the scene."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    origins = [[0.0, 0.0, 0.0]]
    
    prim_utils.create_prim("/World/Origin1", "Xform", translation=origins[0])
    # -- Robot
    emily_cfg = EMILY_CFG.replace(prim_path="/World/Origin1/Robot")
    emily = Articulation(cfg=emily_cfg)
    #ur10_cfg = UR10_CFG.replace(prim_path="/World/Origin1/Robot")
    #ur10_cfg.init_state.pos = (0.0, 0.0, 1.03)
    #ur10 = Articulation(cfg=ur10_cfg)

    # return the scene information
    scene_entities = {
        "emily": emily,
        #"ur10": ur10,
    }
    return scene_entities, origins


def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, Articulation], origins: torch.Tensor):
    """Runs the simulation loop."""
    # Define simulation stepping
    robot = entities["emily"]
    sim_dt = sim.get_physics_dt()
    count = 0
    # Simulate physics
    while simulation_app.is_running():
        # reset
        """
        if count % 500 == 0:
            # reset counters
            count = 0
            # reset the scene entities
            # root state
            # root_state = robot.data.default_root_state.clone()
            # root_state[:, :3] += origins[0]
            # robot.write_root_pose_to_sim(root_state[:, :7])
            # robot.write_root_velocity_to_sim(root_state[:, 7:])
            # set joint positions
            joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            # clear internal buffers
            robot.reset()
            print("[INFO]: Resetting robots state...")
        # apply random actions to the robots
        # perform step
        """
        sim.step()
        # update sim-time
        count += 1
        # update buffers
        robot.update(sim_dt)


def main():
    """Main function."""
    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])
    # design scene
    scene_entities, scene_origins = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene_entities, scene_origins)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
