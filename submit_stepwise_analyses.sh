#!/bin/bash

# Script for stepwise model testing process. 
# 
# It systematically builds up the model complexity:
# 1. Unconditional Growth Models (Time structures)
# 2. Random Effects structures
# 3. Covariates (Controls)

JOB_SCRIPT="submit_baseline.job"
DELAY=2

echo "=================================================================="
echo "Starting Replication of Analysis Steps"
echo "=================================================================="


# --- STEP 1: Time structure ---

# 1. Linear Time, Random Intercept
echo "[1/7] Submitting: Linear Time, Random I."
sbatch "$JOB_SCRIPT" \
    --random 1 \
    --controls "none"
sleep $DELAY

# 2. Quadratic Time, Random Intercept
echo "[2/7] Submitting: Linear + Quadratic Time, Random I."
sbatch "$JOB_SCRIPT" \
    --random 1 \
    --controls "none" \
    -q
sleep $DELAY


# --- STEP 2: Random effects ---

# 3. Quadratic Time, Random Slope (Time)
echo "[3/7] Submitting: Linear + Quadratic Time, Random I. + S. (Time)"
sbatch "$JOB_SCRIPT" \
    --random 2 \
    --controls "none" \
    -q
sleep $DELAY

# 4. Quadratic Time, Random Slope (Time^2)
echo "[4/7] Submitting: Linear + Quadratic Time, Random I. + S. (Time^2)"
sbatch "$JOB_SCRIPT" \
    --random 3 \
    --controls "none" \
    -q
sleep $DELAY


# --- PHASE 3: Adding Covariates (Fixed Effects) ---
# Base config: Quadratic Time (-q) + Full Random Effects (--random 3)

echo "[5/7] Submitting: Linear + Quadratic Time, Random I. + S. (Time^2), Scenario Controls"
sbatch "$JOB_SCRIPT" \
    --random 3 \
    --controls "scenario" \
    -q
sleep $DELAY

echo "[6/7] Submitting: Linear + Quadratic Time, Random I. + S. (Time^2), Experiment Controls"
sbatch "$JOB_SCRIPT" \
    --random 3 \
    --controls "experiment" \
    -q
sleep $DELAY


echo "[7/7] Submitting: Linear + Quadratic Time, Random I. + S. (Time^2), Scenario + Experiment Controls"
sbatch "$JOB_SCRIPT" \
    --random 3 \
    --controls "all" \
    -q
sleep $DELAY

echo "=================================================================="
echo "All jobs submitted. Check outputs/logs/ for progress."
echo "=================================================================="
