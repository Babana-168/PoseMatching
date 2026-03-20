@echo off
setlocal
cd /d "%~dp0"

set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" goto :ERR_VSWHERE

set "VSTMP=%TEMP%\vs_install_path.txt"
"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath > "%VSTMP%"
if errorlevel 1 goto :ERR_VSWHERE_FAILED

set "VSINSTALL="
if exist "%VSTMP%" set /p VSINSTALL=<"%VSTMP%"
if exist "%VSTMP%" del /q "%VSTMP%"
if "%VSINSTALL%"=="" goto :ERR_VSINSTALL

set "VCVARS=%VSINSTALL%\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%VCVARS%" goto :ERR_VCVARS

call "%VCVARS%"
if errorlevel 1 goto :ERR_VCVARS_INIT

echo Compiling...
cl.exe /O2 /fp:fast /openmp /arch:AVX2 /EHsc /std:c++17 /I"C:\opencv\build\include" main.cpp /Fe:pose_match.exe /link /LIBPATH:"C:\opencv\build\x64\vc16\lib" opencv_world4120.lib
if errorlevel 1 goto :BUILD_FAILED

echo === BUILD OK ===
copy /Y "C:\opencv\build\x64\vc16\bin\opencv_world4120.dll" . >nul 2>&1
echo Run: pose_match.exe
endlocal
exit /b 0

:BUILD_FAILED
echo === BUILD FAILED ===
endlocal
exit /b 1

:ERR_VSWHERE
echo ERROR: vswhere.exe not found: %VSWHERE%
endlocal
exit /b 1

:ERR_VSWHERE_FAILED
echo ERROR: vswhere failed to locate Visual Studio.
if exist "%VSTMP%" del /q "%VSTMP%"
endlocal
exit /b 1

:ERR_VSINSTALL
echo ERROR: Visual Studio with C++ tools not found.
echo        Install "Desktop development with C++".
endlocal
exit /b 1

:ERR_VCVARS
echo ERROR: vcvars64.bat not found: %VCVARS%
endlocal
exit /b 1

:ERR_VCVARS_INIT
echo ERROR: Failed to initialize MSVC environment.
endlocal
exit /b 1
