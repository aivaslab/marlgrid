
testEnvs = []
for level in range(3):
    for sublevel in range(levelRanges[level]):
        tempEnv = wrap_env(para_TutorialEnv, width=9, height=9, agents=agents, step_reward=-0.2, done_reward=-1)
        tempEnv.unwrapped.random_mode = False
        tempEnv.unwrapped.mylevel = level+1
        tempEnv.unwrapped.myLevel = sublevel+1
        testEnvs += [VecMonitor(pz2sb3(tempEnv), )]