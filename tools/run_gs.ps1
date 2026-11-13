param([switch]$UseGeer, [Parameter(ValueFromRemainingArguments=$true)][string[]]$PythonArgs)
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vs = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vs) { throw "Visual Studio C++ tools not installed" }
Import-Module "$vs\Common7\Tools\Microsoft.VisualStudio.DevShell.dll"
Enter-VsDevShell -VsInstallPath $vs -SkipAutomaticLocation -DevCmdArguments '-arch=x64 -host_arch=x64' | Out-Null
$venv = if ($UseGeer) { ".venv-geer" } else { ".venv-gs" }
$cache = if ($UseGeer) { "cuda-cache-geer" } else { "cuda-cache" }
$env:PATH = "$root\$venv\Scripts;$env:PATH"
$env:MAX_JOBS = "2"
$env:TORCH_CUDA_ARCH_LIST = "8.6"
$env:TORCH_EXTENSIONS_DIR = "$root\outputs\$cache"
$env:DISTUTILS_USE_SDK = "1"
# CUDA 12.4 predates the installed MSVC 14.44; use NVIDIA's explicit override.
$env:NVCC_PREPEND_FLAGS = "-allow-unsupported-compiler"
& "$root\$venv\Scripts\python.exe" @PythonArgs
exit $LASTEXITCODE
