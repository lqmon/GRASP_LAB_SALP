# SALP Training Debug Report

## Problems Identified

### 1. **Coordinate System Mismatch** (CRITICAL - FIXED)
- **Issue**: Food collection and respawning used mixed coordinate systems
  - `robot_pos`: in METERS (physics frame)
  - `food_positions`: in PIXELS (rendering frame)
  - Distance calculations were comparing meters to pixels, causing nonsensical results

- **Location**: 
  - `_respawn_food()` line ~280: Robot distance check mixed coordinates
  - `_get_extended_observation()` line ~475: Angle calculation mixed coordinates

- **Impact**: Food could never be collected because distance checks were meaningless

- **Fix Applied**:
  ```python
  # BEFORE (Wrong):
  robot_distance = math.sqrt((x - self.robot_pos[0])**2 + (y - self.robot_pos[1])**2)
  # x, y are pixels; robot_pos is meters - WRONG!
  
  # AFTER (Correct):
  robot_pos_pixels = np.array([
      self.pos_init[0] + self.robot_pos[0] * self.pixel_scale,
      self.pos_init[1] + self.robot_pos[1] * self.pixel_scale
  ])
  robot_distance = math.sqrt((x - robot_pos_pixels[0])**2 + (y - robot_pos_pixels[1])**2)
  # Now both are in pixels - CORRECT
  ```

### 2. **Episode Duration** (VERIFIED - NOT ACTUALLY A PROBLEM)
- **Status**: Episodes run for 500 steps, which gives -25.00 reward
- **Math**: 500 steps × (-0.05 time_penalty) = -25.00 ✓
- **Conclusion**: Episode length is CORRECT

### 3. **Zero Food Collection Across 1000 Episodes** (ROOT CAUSE)
- **Why**: Coordinate system bugs prevented food collection detection
- **Expected After Fix**: Food collection should now work properly
- **To Verify**: Run training and check if Food count increases

### 4. **No Learning Progress** (CONSEQUENCE OF #3)
- **Why**: All episodes have identical rewards (-25) because no food is collected
- **Impact**: Agent receives no positive signal to learn from
- **Expected After Fix**: Once food collection works, agent should see variable rewards

### 5. **Slow Physics Simulation** (PERFORMANCE ISSUE)
- **Issue**: Current physics-based robot simulation is computationally expensive
  - Each episode takes 1-2 minutes
  - 1000 episodes would take 16-33 hours
- **Root Cause**: `robot.step_through_cycle()` involves complex physics calculations
- **Workaround**: Use `--sb3` flag to use Stable Baselines3 implementation instead

## Files Modified

1. **src/salp/environments/salp_snake_env.py**
   - Fixed `_respawn_food()` coordinate system (line ~280)
   - Fixed `_get_extended_observation()` coordinate system (line ~475)
   - Added coordinate system comments for clarity

## Testing Recommendations

### Quick Test (5-10 minutes):
```bash
python3 train.py --config sb3_fast --sb3
```
- Uses Stable Baselines3 SAC (faster than custom implementation)
- Smaller environment for faster physics
- Should complete ~50 episodes in 5-10 minutes
- Check if Food count increases in logs

### Full Test (2-3 hours):
```bash
python3 train.py --config defaults
```
- Uses custom trainer
- Full-sized environment
- Should show improvement over 1000 episodes now that food can be collected

## Expected Behavior After Fixes

**Before Fixes**:
```
Episode 1/1000, Score: -25.00, Food: 0, Steps: 500
Episode 2/1000, Score: -25.00, Food: 0, Steps: 500
...all identical...
Episode 1000/1000, Score: -25.00, Food: 0, Steps: 500
```

**After Fixes**:
```
Episode 1/1000, Score: -15.00, Food: 1, Steps: 285  # Food collected!
Episode 2/1000, Score: -22.50, Food: 0, Steps: 450
Episode 3/1000, Score: -8.50, Food: 2, Steps: 150   # Better performance!
...scores should vary...
```

## Next Steps

1. **Verify Food Collection Works**
   - Run training with one of the test configs
   - Monitor Food column in output
   - Should see value > 0 in some episodes

2. **Optimize Speed (Optional)**
   - Use `--sb3` for faster training
   - Reduce environment size
   - Use `--config sb3_fast` template

3. **Improve Learning**
   - Tune reward shaping (food_reward, collision_penalty)
   - Adjust exploration parameters
   - Monitor loss metrics in training.log

## Summary

The main issue preventing learning was the coordinate system mismatch that broke food collection detection. With this fix applied, the agent should now:
- Receive positive rewards when collecting food
- Have varying episode rewards
- Be able to learn behaviors to maximize food collection

The slow physics simulation is a separate performance issue that can be addressed with the SB3 backend if needed.
