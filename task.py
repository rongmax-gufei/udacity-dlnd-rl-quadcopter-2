import numpy as np
from physics_sim import PhysicsSim


class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""

    def __init__(self, init_pose=None, init_velocities=None,
                 init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime)
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.])

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        # reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()

        # return reward
        reward = 0.0

        # Reward positive velocity along z-axis
        reward += self.sim.v[2]

        # Reward positions close to target along z-axis
        reward -= (abs(self.sim.pose[2] - self.target_pos[2])) / 2.0

        # A lower sensativity towards drifting in the xy-plane
        reward -= (abs(self.sim.pose[:2] - self.target_pos[:2])).sum() / 4.0

        reward -= (abs(self.sim.angular_v[:3])).sum()





        # reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        # reward = 1.-.3*(abs(self.sim.pose[2] - self.target_pos[2]))
        # reward = 1.-.3*(np.linalg.norm(self.sim.pose[:3] - self.target_pos)).sum()

        # Scale reward to [-1, 1]
        z_distance = abs(self.sim.pose[2] - self.target_pos[2]) / self.max_z_distance  # [0, 1]
        reward = - (z_distance * 2)  # - abs(self.sim.v[2]) * 0.01
        # print('z={:3.2f}, z_distance={:3.2f}, reward={:3.2f}'.format(self.sim.pos[2], z_distance, reward))





        distance_to_target = np.linalg.norm(self.target_pos - self.sim.pose[:3])
        sum_acceleration = np.linalg.norm(self.sim.linear_accel)
        reward = (5. - distance_to_target) * 0.3 - sum_acceleration * 0.05

        # Positional error
        # loss = np.linalg.norm((self.sim.pose[:3] - self.target_pos))

        # reward = 1/(1+penalty)
        # return np.clip(1-(self.sim.pose[2]-self.target_pos[2])**2, 0, 1)

        # return self.reward_from_huber_loss(penalty, delta=1, max_reward=1, min_reward=0)
        loss = (self.sim.pose[2] - self.target_pos[2]) ** 2
        loss += 0.1 * self.sim.linear_accel[2] ** 2
        reward = self.reward_from_huber_loss(loss, delta=0.5)
        return reward

    def reward_from_huber_loss(self, x, delta, max_reward=1, min_reward=0):
        return np.maximum(max_reward - delta * delta * (np.sqrt(1 + (x / delta) ** 2) - 1), min_reward)


        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds)  # update the sim pose and velocities
            reward += self.get_reward()
            pose_all.append(self.sim.pose)
            #
            if self.sim.pose[2] >= self.target_pos[2]:
                reward += 100
                done = True
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat)
        return state
