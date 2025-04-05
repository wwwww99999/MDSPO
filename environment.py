def get_velocity_threshold(env_name):
    thresholds = {'SafetyHopperVelocity-v1': 0.7402,
                  'SafetyAntVelocity-v1': 2.6222,
                  'SafetyHumanoidVelocity-v1': 1.4149,
                  'SafetyWalker2dVelocity-v1': 2.3415,
                  'SafetyHalfCheetahVelocity-v1': 3.2096,
                  'SafetySwimmerVelocity-v1': 0.2282}
    return thresholds[env_name]