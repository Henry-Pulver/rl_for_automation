import gym


class MountainCarEnv(gym.Env):
    def __init__(self, max_episode_length: int):
        super(MountainCarEnv, self).__init__("MountainCar-v0")
        self.max_ep_len = max_episode_length
        self.step = 0

    def step(self, action):
        obs, rew, done, info = super(MountainCarEnv, self).step(action=action)
        self.step += 1
        # If solves mountain car problem successfully
        if done:
            rew = 1000
            self.step = 0
        else:
            rew = 0
        if self.step >= self.max_ep_len:
            done = True
        return obs, rew, done, info
