# Quick Reference: Running Improved Training

## Basic Commands

### Start training (SB3 with default config):
```bash
python train.py --sb3
```

### Watch progress in real-time:
```bash
tail -f training.log
```

### View latest configuration:
```bash
cat data/models/$(ls -t data/models | grep salp_training | head -1)/config.json | jq '.'
```

### Check episode statistics:
```bash
ls -la data/models/*/logs/ | head -20
cat data/models/*/logs/episode_100.json | jq '.'
```

## What Changed in train.py

### Before:
- Minimal error handling
- Limited logging
- Silent failures
- No checkpoint validation
- Difficult to debug issues

### After:
- ✅ Comprehensive error handling with stack traces
- ✅ Full logging to file and console
- ✅ Argument and path validation
- ✅ Environment sanity checks
- ✅ Configuration saved with each run
- ✅ Episode-level statistics tracking
- ✅ Progress monitoring
- ✅ Interrupt handling with model saving

## Troubleshooting

### "Checkpoint file not found"
```bash
# Check available checkpoints
ls -la data/models/*/checkpoints/

# Use correct path
python train.py --checkpoint data/models/salp_training_YYYYMMDD_HHMMSS/checkpoints/checkpoint_NNNNN_steps.zip
```

### "Configuration file not found"
```bash
# List available configs
ls configs/

# Use correct config name
python train.py --config single_food --sb3
```

### Debug training issues:
```bash
# Enable debug logging
python train.py --sb3 --debug --timesteps 1000

# Check training.log for detailed diagnostics
cat training.log | grep -i error
```

## Key Features Added

1. **Logging**: All events logged to `training.log` and console
2. **Validation**: Arguments and paths checked before training
3. **Environment Tests**: Verifies environment works before training
4. **Config Saving**: Each run saves config.json for reproducibility
5. **Episode Tracking**: Per-episode statistics saved to JSON
6. **Error Messages**: Helpful errors with suggestions
7. **Interrupt Handling**: Saves model on Ctrl+C
8. **Status Indicators**: Visual feedback (✓ ✗ 🚀 🏆)

## Why These Improvements Matter

When training RL models, it's crucial to:
- **Know what's happening** → Logging provides visibility
- **Catch errors early** → Validation prevents wasted compute
- **Reproduce runs** → Saved configs enable reproducibility
- **Track progress** → Episode stats guide debugging
- **Handle interruptions** → Graceful saves prevent lost work
- **Debug efficiently** → Stack traces speed up problem-solving

## Next: Fix Environment Issues

The logs show episodes terminating after 1 step. To investigate:

```bash
# Check environment independently
python -c "
from src.salp.environments.salp_snake_env import SalpSnakeEnv
env = SalpSnakeEnv(render_mode=None, num_food_items=5)
obs, info = env.reset()
print(f'Initial observation shape: {obs.shape}')
for i in range(10):
    obs, r, term, trunc, info = env.step(env.action_space.sample())
    print(f'Step {i}: reward={r:.2f}, done={term}, info={info}')
    if term:
        break
"
```

This will help identify if the issue is in:
- Environment initialization
- Collision detection logic
- Action rescaling
- Observation shape mismatch
