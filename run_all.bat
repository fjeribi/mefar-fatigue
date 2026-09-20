@echo off
REM Reproduce every analysis in the paper (Windows).
REM Usage: run_all.bat "C:\path\to\MEFAR" "C:\path\to\MEFAR_DOWN.csv" [results_dir]
set RAW_ROOT=%~1
set RELEASE=%~2
set OUT=%~3
if "%OUT%"=="" set OUT=results
python scripts\01_protocol1_sample_level.py --release "%RELEASE%" --out-dir "%OUT%" || exit /b 1
python scripts\02_reconstruct_subjectwise.py --raw-root "%RAW_ROOT%" --out-dir "%OUT%" || exit /b 1
python scripts\03_protocol2_conventional.py --out-dir "%OUT%" || exit /b 1
python scripts\04_protocol2_deep.py --out-dir "%OUT%" || exit /b 1
python scripts\05_session_proxy.py --out-dir "%OUT%" || exit /b 1
python scripts\06_ablation_statistics.py --out-dir "%OUT%" || exit /b 1
python scripts\07_gating_and_masking.py --out-dir "%OUT%" || exit /b 1
python scripts\08_partitioning_control.py --out-dir "%OUT%" || exit /b 1
python scripts\09_descriptives.py --out-dir "%OUT%" --release "%RELEASE%" --raw-root "%RAW_ROOT%" || exit /b 1
python scripts\10_uncertainty.py --out-dir "%OUT%" || exit /b 1
python scripts\11_windowed_features.py --raw-root "%RAW_ROOT%" --out-dir "%OUT%" || exit /b 1
echo All analyses complete. Results in %OUT%\
