import configparser

import Actor
import Critic
from SimWorlds.SimWorld import SimWorld


class CATT:
    """
    Class connecting Actor-Critic and Environment.
    Uses Table Critic
    """
    def __init__(self, sim_world: SimWorld):
        config = configparser.ConfigParser()
        config.read('config.ini')
        self.actor = Actor.Actor()
        self.critic = Critic.Critic()
        self.sim_world = sim_world
        self.max_episode_steps = config["PRIMARY"].getint('MAX_EPISODE_STEPS')
        self.max_episodes = config["PRIMARY"].getint('MAX_EPISODES')

    def rl_loop(self):
        episode_success = {}
        for j in range(self.max_episodes):
            self.actor.reset_e()
            self.critic.reset_e()
            state = self.sim_world.produce_initial_state()
            print(f"Episode # {j}")
            possible_actions = self.sim_world.get_legal_actions()
            action = self.actor.get_action(state, possible_actions)

            current_episode_sa = {(state, action)}
            if j == self.max_episodes - 1:
                # Get best actions for final episode
                self.actor.max_greed()
            t = 0
            while t < self.max_episode_steps:
                state_next, reward = self.sim_world.apply_action(action)
                possible_actions = self.sim_world.get_legal_actions()
                action_next = self.actor.get_action(state_next, possible_actions)
                self.actor.set_eligibility(state, action, 1)
                self.critic.update_td_error(reward, state, state_next)
                self.critic.set_eligibility(state, 1)

                self.critic.update_state_action(current_episode_sa)
                self.critic.update_e(current_episode_sa)
                self.actor.update_state_action(current_episode_sa, self.critic.get_td_error())
                self.actor.update_e(current_episode_sa)

                state = state_next
                action = action_next
                current_episode_sa.add((state, action))
                """ Break out of episode if current state is final"""
                t += 1
                if self.sim_world.is_current_state_final():
                    break

            if j == self.max_episodes - 1:
                self.sim_world.plot_world_state()

            episode_success[j] = t
            self.sim_world.reset_world()

            if j % self.max_episodes / 10 == 0:  # This ensures a nice epsilon degradation. If it begins at 0.5, it ends at 0.001
                self.actor.update_epsilon()

        return self.actor.policy, episode_success

