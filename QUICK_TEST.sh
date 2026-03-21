#!/bin/bash
# Quick reference commands for testing the SALP training fixes

echo "SALP Training - Test Commands"
echo "=============================="
echo ""

echo "1. QUICK TEST (with Stable Baselines3 - much faster):"
echo "   cd /Users/puneet/GRASP_LAB_SALP"
echo "   python3 train.py --config sb3_fast --sb3"
echo "   -> Should complete in 10-15 minutes"
echo "   -> Look for 'Food' column to be > 0 (was always 0 before fix)"
echo ""

echo "2. STANDARD TEST (with custom physics):"
echo "   python3 train.py --config defaults"
echo "   -> Will take longer (~2 hours for 1000 episodes)"
echo "   -> Should now see food collection and learning"
echo ""

echo "3. VIEW LATEST LOG:"
echo "   tail -100 training.log"
echo ""

echo "4. VIEW DETAILED DEBUG REPORT:"
echo "   cat DEBUG_REPORT.md"
echo ""

echo "5. VIEW FIX SUMMARY:"
echo "   cat FIX_SUMMARY.md"
echo ""

echo "EXPECTED CHANGES AFTER FIX:"
echo "  ❌ BEFORE: Food: 0, Score: -25.00 (every episode identical)"
echo "  ✅ AFTER:  Food: 1, Score: -15.00 (varying scores per episode)"
echo ""
