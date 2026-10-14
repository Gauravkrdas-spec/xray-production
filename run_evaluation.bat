@echo off
REM RadVision AI - full re-evaluation on IU X-Ray (both ablation arms + statistics)
REM Usage:  run_evaluation.bat "C:\path\to\iu_xray"
REM The folder must contain images\images_normalized\, indiana_projections.csv, indiana_reports.csv
REM backend\.env must contain ANTHROPIC_API_KEY=...
setlocal
if "%~1"=="" (echo Usage: run_evaluation.bat "C:\path\to\iu_xray" & exit /b 1)
set DATA=%~1
cd /d "%~dp0backend"
python evaluation_pipeline.py --location prior   --tag without_gradcam --data-dir "%DATA%" || exit /b 1
python evaluation_pipeline.py --location gradcam --tag with_gradcam    --data-dir "%DATA%" || exit /b 1
cd /d "%~dp0"
python analysis\statistics.py evaluation_results_v2\without_gradcam evaluation_results_v2\with_gradcam
endlocal
