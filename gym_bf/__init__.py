from gym.envs.registration import register

register(
    id='bf-v0',
    entry_point='gym_bf.envs:BFEnv',
    kwargs = {"space_seed":0,"n":10}
)