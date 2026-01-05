# Model Improvement Proposal

## Current Performance Issues

### Critical Findings:
1. **High-scoring games severely under-predicted**: >25 pts → predicts 22.10, actual 31.44 (bias: -9.34)
2. **Low-scoring games over-predicted**: <5 pts → predicts 5.70, actual 1.75 (bias: +3.95)
3. **Tier 3 (stars) has 36.9% large errors** (>8 points)
4. **15.5% of all games have large errors** (>8 points)

---

## Proposed Improvements (Ranked by Impact)

### 🎯 **TIER 1: HIGHEST IMPACT** (Expected 0.3-0.5 MAE improvement)

#### 1. **Player-Specific vs Opponent History** ⭐⭐⭐
**What**: Track how each player performs vs each specific opponent team
**Why**: Some players have strong/weak matchups vs specific teams (e.g., LeBron vs Spurs)
**Features**:
- `player_vs_opp_pts_avg_last_5` - Player's avg points vs this opponent (last 5 meetings)
- `player_vs_opp_pts_avg_career` - Career avg vs this opponent
- `player_vs_opp_minutes_avg_last_5` - Minutes vs this opponent
- `player_vs_opp_games_count` - Number of times faced this opponent

**Expected Impact**: 0.2-0.4 MAE improvement, especially for stars

---

#### 2. **Game Script / Blowout Probability** ⭐⭐⭐
**What**: Predict likelihood of blowout (affects garbage time, bench minutes)
**Why**: Model under-predicts high scores (31.44 actual vs 22.10 predicted) - likely blowout games
**Features**:
- `blowout_prob` - Probability game is decided by >15 points (from Vegas spread)
- `garbage_time_minutes_est` - Estimated garbage time minutes
- `is_likely_blowout` - Binary: spread >12 or team margin trend suggests blowout
- `vegas_spread_abs_normalized` - Normalized absolute spread (0-1 scale)

**Expected Impact**: 0.2-0.3 MAE improvement, especially for high-scoring games

---

#### 3. **Opponent Defensive Rating by Position (Enhanced DvP)** ⭐⭐
**What**: More granular opponent defense metrics
**Why**: Current DvP is basic; need opponent's defensive efficiency vs position
**Features**:
- `opp_def_rating_vs_pos` - Opponent's defensive rating vs this position (points per 100 possessions)
- `opp_fg_pct_allowed_vs_pos` - FG% allowed vs position
- `opp_3pt_pct_allowed_vs_pos` - 3PT% allowed vs position
- `opp_pace_vs_pos` - Pace when defending this position

**Expected Impact**: 0.1-0.2 MAE improvement

---

### 🎯 **TIER 2: MEDIUM IMPACT** (Expected 0.1-0.3 MAE improvement)

#### 4. **Lineup Context / Rotation Depth** ⭐⭐
**What**: Who else is playing? Is the team shorthanded?
**Why**: Affects usage and minutes distribution
**Features**:
- `teammate_injuries_count` - Number of key teammates injured
- `team_rotation_depth` - Available players at this position
- `is_shorthanded` - Binary: team missing >2 key players
- `teammate_usage_available` - Sum of injured teammates' usage (more usage available)

**Expected Impact**: 0.1-0.2 MAE improvement

---

#### 5. **Time-of-Season Context** ⭐
**What**: Playoff push, tanking, end-of-season rest
**Why**: Affects motivation and minutes
**Features**:
- `games_remaining` - Games left in season
- `playoff_prob` - Team's playoff probability (from Elo/standings)
- `is_playoff_push` - Binary: team fighting for playoff spot
- `is_tanking` - Binary: team clearly out of playoffs
- `is_end_of_season` - Binary: last 10 games of season

**Expected Impact**: 0.1-0.15 MAE improvement

---

#### 6. **Recent Form vs Similar Opponents** ⭐
**What**: How has player performed vs teams with similar defensive style?
**Why**: Some players match up better vs certain defensive schemes
**Features**:
- `pts_vs_similar_def_last_10` - Points vs teams with similar defensive rating
- `opp_def_style_match` - Match score: how similar is opponent's defense to player's preferred style

**Expected Impact**: 0.05-0.1 MAE improvement

---

### 🎯 **TIER 3: ARCHITECTURAL IMPROVEMENTS** (Expected 0.1-0.2 MAE improvement)

#### 7. **Quantile Regression for High/Low Scores** ⭐⭐
**What**: Separate models for predicting high-scoring games (>25) and low-scoring games (<5)
**Why**: Current model has systematic bias (under-predicts high, over-predicts low)
**Approach**:
- Train separate models for different score ranges
- Use ensemble: `pred = w1*high_model + w2*base_model + w3*low_model`
- Or use quantile regression (predict 10th, 50th, 90th percentiles)

**Expected Impact**: 0.2-0.3 MAE improvement, especially for outliers

---

#### 8. **Better Sigma Model (Heteroscedastic Variance)** ⭐
**What**: Improve variance prediction for probability calibration
**Why**: Current sigma model is basic; stars have much higher variance
**Features**:
- Add player tier to sigma model
- Add opponent defensive variance
- Add game script variance (blowouts = low variance)

**Expected Impact**: Better probability estimates, 0.05-0.1 MAE improvement

---

#### 9. **Ensemble of Tiered Models** ⭐
**What**: Combine unified + tiered models with weights
**Why**: Unified model may be better for some cases, tiered for others
**Approach**:
- Train both unified and tiered models
- Use weighted average: `pred = 0.6*tiered + 0.4*unified`
- Or use tiered for stars, unified for bench

**Expected Impact**: 0.05-0.1 MAE improvement

---

## Implementation Priority

### Phase 4A (Quick Wins - 1-2 weeks):
1. **Player vs Opponent History** - High impact, straightforward to implement
2. **Game Script Features** - High impact, uses existing Vegas data
3. **Enhanced DvP** - Medium impact, extends existing feature

**Expected Combined Impact**: 0.4-0.7 MAE improvement

### Phase 4B (Medium-term - 2-4 weeks):
4. **Lineup Context** - Requires injury data integration (already have)
5. **Time-of-Season Context** - Requires standings/Elo data
6. **Quantile Regression** - Architectural change

**Expected Combined Impact**: Additional 0.2-0.4 MAE improvement

### Phase 4C (Long-term - 1-2 months):
7. **Better Sigma Model** - Improves probability calibration
8. **Ensemble Methods** - Fine-tuning

**Expected Combined Impact**: Additional 0.1-0.2 MAE improvement

---

## Total Expected Improvement

**Conservative Estimate**: 0.5-0.8 MAE improvement (11-18% better)
**Optimistic Estimate**: 0.7-1.2 MAE improvement (15-26% better)

**Current MAE**: 4.530
**Target MAE**: 3.5-4.0 (22-30% improvement)

---

## Recommendation

**Start with Phase 4A** - These three features are:
- High impact (0.4-0.7 MAE improvement expected)
- Relatively easy to implement
- Use existing data sources
- Address the biggest issues (high-scoring under-prediction, opponent matchups)

**Most Important**: Player vs Opponent History - This directly addresses the fact that some players consistently perform better/worse vs specific teams, which the model currently cannot capture.

