# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import json
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers import VisualizationMarkers
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms

##
# Pre-defined configs
##
from isaaclab_assets import CRAZYFLIE_CFG  # isort: skip
from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip


class QuadcopterEnvWindow(BaseEnvWindow):
    """Window manager for the Quadcopter environment."""

    def __init__(self, env: QuadcopterEnv, window_name: str = "IsaacLab"):
        """Initialize the window.

        Args:
            env: The environment object.
            window_name: The name of the window. Defaults to "IsaacLab".
        """
        # initialize base window
        super().__init__(env, window_name)
        # add custom UI elements
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    # add command manager visualization
                    self._create_debug_vis_ui_element("targets", self.env)


@configclass
class QuadcopterEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 10.0
    decimation = 2
    action_space = 4
    observation_space = 12
    state_space = 0
    debug_vis = True

    ui_window_class_type = QuadcopterEnvWindow

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=True)

    # robot
    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    config_path  = '/home/miller/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/quadcopter/default_config.json'

    # reward scales
    lin_vel_reward_scale = -0.05
    ang_vel_reward_scale = -0.01
    distance_to_goal_reward_scale = 15.0


class QuadcopterEnv(DirectRLEnv):
    cfg: QuadcopterEnvCfg

    def __init__(self, cfg: QuadcopterEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        params = json.load(open(self.cfg.config_path))

        # Total thrust and moment applied to the base of the quadcopter
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        # Goal position
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "lin_vel",
                "ang_vel",
                "distance_to_goal",
            ]
        }
        # Get specific body indices
        self._body_id = self._robot.find_bodies("body")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._root_body_inertia_mat = self._robot.root_physx_view.get_inertias()[0][0].to(self.device)
        self._desired_mass = params['mass']
        self.mass_correction = self._robot_mass / self._desired_mass
        self._desired_inertias = torch.tensor(params['inertia_diag'], device=self.device)
        self.inertia_correction = torch.diag(self._root_body_inertia_mat.reshape((3,3)))/ self._desired_inertias
        # self._robot.root_physx_view.set_inertias(self._inertias,torch.tensor([0]))
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

        self._thrust_coefficients = torch.tensor(params['thrust_coefficients'], device=self.device)
        self._thrust_directions = torch.tensor(params['rotor_thrust_directions'], dtype=torch.float32, device=self.device)
        self._rotor_torque_directions = torch.tensor(params['rotor_torque_directions'], dtype=torch.float32, device=self.device)
        self._rotor_torque_constants = torch.tensor(params['rotor_torque_constants'], dtype=torch.float32, device=self.device)
        self._rotor_positions = torch.tensor(params['rotor_positions'], dtype=torch.float32, device=self.device)
        self._rising_delay_constants = 1/torch.tensor(params['delay_rising_constants'], dtype=torch.float32, device=self.device)
        self._falling_delay_constants = 1/torch.tensor(params['delay_falling_constants'], dtype=torch.float32, device=self.device)
        self._rotor_speeds = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone().clamp(-1.0, 1.0)
        actions_0_1 = (self._actions + 1.0) / 2.0

        # apply motor delay
        rising_mask = actions_0_1 > self._rotor_speeds
        falling_mask = actions_0_1 <= self._rotor_speeds
        diffs = actions_0_1 - self._rotor_speeds
        self._rotor_speeds[rising_mask] += (diffs* self._rising_delay_constants)[rising_mask]  * self.cfg.sim.dt
        self._rotor_speeds[falling_mask] += (diffs * self._falling_delay_constants)[falling_mask]* self.cfg.sim.dt

        # step 1: quadratic thrust curve
        actions_polyomial = torch.vstack([
            torch.ones_like(self._rotor_speeds)[torch.newaxis, :], 
            self._rotor_speeds[torch.newaxis, :], 
            torch.square(self._rotor_speeds)[torch.newaxis, :]
        ]) # 3 x N x 4
        thrust_magnitude = actions_polyomial @ self._thrust_coefficients
        thrust_magnitude = torch.einsum('kij,jk->ij', actions_polyomial, self._thrust_coefficients) # N x 4
        rotor_thrust = thrust_magnitude[...,torch.newaxis] * self._thrust_directions[torch.newaxis,...]

        # yaw moment (torque in z axis)
        torque =(thrust_magnitude * self._rotor_torque_constants) @ self._rotor_torque_directions
        # roll and pitch moment (torque in x and y axis)
        cross_prod = sum([
            torch.cross(self._rotor_positions[i].expand(rotor_thrust[:,i,:].shape), rotor_thrust[:,i,:])
            for i in range(4)
        ])
        torque += cross_prod

        # index 0 is the body (indices 1-4 are the rotors)
        self._thrust[:,0,:] = rotor_thrust.sum(dim=1) * self.mass_correction # dirty hack to get the dynamics right since editing the mass/inertia seems to not work
        self._moment[:,0,:] = torque * self.inertia_correction 

    def _apply_action(self):
        self._robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)

    def _get_observations(self) -> dict:
        desired_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_pos_w, self._robot.data.root_quat_w, self._desired_pos_w
        )
        obs = torch.cat(
            [
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
                self._robot.data.projected_gravity_b,
                desired_pos_b,
            ],
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        lin_vel = torch.sum(torch.square(self._robot.data.root_lin_vel_b), dim=1)
        ang_vel = torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)
        distance_to_goal = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_pos_w, dim=1)
        distance_to_goal_mapped = 1 - torch.tanh(distance_to_goal / 0.8)
        rewards = {
            "lin_vel": lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt,
            "ang_vel": ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt,
            "distance_to_goal": distance_to_goal_mapped * self.cfg.distance_to_goal_reward_scale * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = torch.logical_or(self._robot.data.root_pos_w[:, 2] < 0.1, self._robot.data.root_pos_w[:, 2] > 2.0)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # Logging
        final_distance_to_goal = torch.linalg.norm(
            self._desired_pos_w[env_ids] - self._robot.data.root_pos_w[env_ids], dim=1
        ).mean()
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        extras["Metrics/final_distance_to_goal"] = final_distance_to_goal.item()
        self.extras["log"].update(extras)

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0
        # Sample new commands
        self._desired_pos_w[env_ids, :2] = torch.zeros_like(self._desired_pos_w[env_ids, :2]).uniform_(-2.0, 2.0)
        self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
        self._desired_pos_w[env_ids, 2] = torch.zeros_like(self._desired_pos_w[env_ids, 2]).uniform_(0.5, 1.5)
        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first tome
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
                # -- goal pose
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
            # set their visibility to true
            self.goal_pos_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # update the markers
        self.goal_pos_visualizer.visualize(self._desired_pos_w)

#     constexpr Dynamics<T, TI, 4> crazyflie = {
#             // Rotor positions
#             {
#                     {
#                             0.028,
#                             -0.028,
#                             0
#                     },
#                     {
#                             -0.028,
#                             -0.028,
#                             0
#                     },
#                     {
#                             -0.028,
#                             0.028,
#                             0
#                     },
#                     {
#                             0.028,
#                             0.028,
#                             0
#                     },
#             },
#             // Rotor thrust directions
#             {
#                     {0, 0, 1},
#                     {0, 0, 1},
#                     {0, 0, 1},
#                     {0, 0, 1},
#             },
#             // Rotor torque directions
#             {
#                     {0, 0, -1},
#                     {0, 0, +1},
#                     {0, 0, -1},
#                     {0, 0, +1},
#             },
#             // thrust constants
#             {
#                     // {0.0213, -0.0112, 0.1201},
#                     // {0.0213, -0.0112, 0.1201},
#                     // {0.0213, -0.0112, 0.1201},
#                     // {0.0213, -0.0112, 0.1201}
#                     // {0, 0, 0.1302},
#                     // {0, 0, 0.1302},
#                     // {0, 0, 0.1302},
#                     // {0, 0, 0.1302}
#                     {0.00352526, 0.01437313, 0.09223048},
#                     {0.00352526, 0.01437313, 0.09223048},
#                     {0.00352526, 0.01437313, 0.09223048},
#                     {0.00352526, 0.01437313, 0.09223048},
#             },
#             // torque constant
#             {4.665e-3, 4.665e-3, 4.665e-3, 4.665e-3},
#             // T, RPM time constant
#             { // rising: ~(0.040 - 0.080)s from manufacturer plot
#                     0.05545454545454546,
#                     0.05545454545454546,
#                     0.05545454545454546,
#                     0.05545454545454546
#             },
#             { // falling: ~0.398s from manufacturer plot
#                     0.24939393939393945,
#                     0.24939393939393945,
#                     0.24939393939393945,
#                     0.24939393939393945
#             },
#             // mass vehicle
#             0.027 + 0.0017 + 0.0003 + 0.0016, // take-off-weight, sd card deck, sd card, optical flow deck (v2)
#             // gravity
#             {0, 0, -9.81},
#             // J
#             {
#                     {
#                             9.416556729130406e-06,
#                             0.0,
#                             0.0
#                     },
#                     {
#                             0.0,
#                             9.644051701582312e-06,
#                             0.0
#                     },
#                     {
#                             0.0,
#                             0.0,
#                             1.745951732253285e-05
#                     }
#             },
#             // J_inv
#             {
#                     {
#                             106195.93007988465,
#                             0.0,
#                             0.0
#                     },
#                     {
#                             0.0,
#                             103690.85846314249,
#                             0.0
#                     },
#                     {
#                             0.0,
#                             0.0,
#                             57275.35197719487
#                     }
#             },
#             // hovering throttle (julia): sqrt((mass * 9.81/4 - thrust_curve[1])/thrust_curve[3]),
# //            "hovering_throttle": 14475.809152959684,
#             0.7261389721508553, // "hovering_throttle_relative"
#             // action limit
#             {0, 1},
#     };


        # T thrust[3];
        # T torque[3];
        # thrust[0] = 0;
        # thrust[1] = 0;
        # thrust[2] = 0;
        # torque[0] = 0;
        # torque[1] = 0;
        # torque[2] = 0;
        # // flops: N*23 => 4 * 23 = 92
        # for(typename DEVICE::index_t i_rotor = 0; i_rotor < 4; i_rotor++){
        #     // flops: 3 + 1 + 3 + 3 + 3 + 4 + 6 = 23
        #     T rpm = action[i_rotor];
        #     T thrust_magnitude = params.dynamics.rotor_thrust_coefficients[i_rotor][0] + params.dynamics.rotor_thrust_coefficients[i_rotor][1] * rpm + params.dynamics.rotor_thrust_coefficients[i_rotor][2] * rpm * rpm;
        #     T rotor_thrust[3];
        #     rl_tools::utils::vector_operations::scalar_multiply<DEVICE, T, 3>(params.dynamics.rotor_thrust_directions[i_rotor], thrust_magnitude, rotor_thrust);
        #     rl_tools::utils::vector_operations::add_accumulate<DEVICE, T, 3>(rotor_thrust, thrust);

        #     rl_tools::utils::vector_operations::scalar_multiply_accumulate<DEVICE, T, 3>(params.dynamics.rotor_torque_directions[i_rotor], thrust_magnitude * params.dynamics.rotor_torque_constants[i_rotor], torque);
        #     rl_tools::utils::vector_operations::cross_product_accumulate<DEVICE, T>(params.dynamics.rotor_positions[i_rotor], rotor_thrust, torque);
        # }

        # // linear_velocity_global
        # state_change.position[0] = state.linear_velocity[0];
        # state_change.position[1] = state.linear_velocity[1];
        # state_change.position[2] = state.linear_velocity[2];

        # // angular_velocity_global
        # // flops: 16
        # quaternion_derivative<DEVICE, T>(state.orientation, state.angular_velocity, state_change.orientation);

        # // linear_acceleration_global
        # // flops: 21
        # rotate_vector_by_quaternion<DEVICE, T>(state.orientation, thrust, state_change.linear_velocity);
        # // flops: 4
        # rl_tools::utils::vector_operations::scalar_multiply<DEVICE, T, 3>(state_change.linear_velocity, 1 / params.dynamics.mass);
        # rl_tools::utils::vector_operations::add_accumulate<DEVICE, T, 3>(params.dynamics.gravity, state_change.linear_velocity);

        # T vector[3];
        # T vector2[3];

        # // angular_acceleration_local
        # // flops: 9
        # rl_tools::utils::vector_operations::matrix_vector_product<DEVICE, T, 3, 3>(params.dynamics.J, state.angular_velocity, vector);
        # // flops: 6
        # rl_tools::utils::vector_operations::cross_product<DEVICE, T>(state.angular_velocity, vector, vector2);
        # rl_tools::utils::vector_operations::sub<DEVICE, T, 3>(torque, vector2, vector);
        # // flops: 9
        # rl_tools::utils::vector_operations::matrix_vector_product<DEVICE, T, 3, 3>(params.dynamics.J_inv, vector, state_change.angular_velocity);