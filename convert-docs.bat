@echo off
setlocal enabledelayedexpansion

:: Set base directories
set "BASEDIR=%CD%"
set "INPUTDIR=%BASEDIR%\sources"
set "OUTPUTDIR=%BASEDIR%\output"
set "TEXRESOURCESDIR=%BASEDIR%\resources\tex"
set "FILTERSDIR=%BASEDIR%\resources\filters"


:: Verify critical directories and files exist
if not exist "%INPUTDIR%" (
    echo Error: Input directory not found: %INPUTDIR%
    exit /b 1
)

if not exist "%TEXRESOURCESDIR%" (
    echo Error: Style directory not found: %TEXRESOURCESDIR%
    exit /b 1
)

:: Create output directory if it doesn't exist
if not exist "%OUTPUTDIR%" mkdir "%OUTPUTDIR%"

:: Check if pandoc is installed
where pandoc >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Error: Pandoc is not installed or not in PATH
    exit /b 1
)

:: Check if a command was provided
if "%~1"=="" goto :help
if "%~1"=="help" goto :help
if "%~1"=="pdf" goto :pdf
if "%~1"=="all" goto :all
goto :help

:help
echo.
echo Windows Batch Script for the Markdown thesis
echo.
echo Usage:
echo    convert-docs.bat help               show this help message
echo    convert-docs.bat pdf                generate a PDF file
echo    convert-docs.bat all                generate all formats
echo.
echo Get local templates with: pandoc -D latex/html/etc
echo or generic ones from: https://github.com/jgm/pandoc-templates
goto :eof

:pdf
echo Generating PDF...
:: Check for required files
if not exist "%TEXRESOURCESDIR%\template.tex" (
    echo Error: template.tex not found in %TEXRESOURCESDIR%
    exit /b 1
)
if not exist "%TEXRESOURCESDIR%\preamble.tex" (
    echo Error: preamble.tex not found in %TEXRESOURCESDIR%
    exit /b 1
)

:: Use type command to verify markdown files exist
set "MD_FILES="
for %%f in ("%INPUTDIR%\*.md") do (
    set "MD_FILES=!MD_FILES! "%%f""
)
if "%MD_FILES%"=="" (
    echo Error: No markdown files found in %INPUTDIR%
    exit /b 1
)

pandoc %MD_FILES% ^
    "%INPUTDIR%\metadata.yml" ^
    --output="%OUTPUTDIR%\Localization-Report.pdf" ^
    --listings ^
    --include-in-header="%TEXRESOURCESDIR%\listings.tex" ^
    --variable=fontsize:12pt ^
    --variable=geometry:margin=1in ^
    --variable=papersize:a4 ^
    --variable=documentclass:report ^
    --lua-filter="%FILTERSDIR%\figure-short-captions.lua" ^
    --lua-filter="%FILTERSDIR%\table-short-captions.lua" ^
    --pdf-engine=xelatex ^
    --number-sections ^
    --verbose ^
    2>"%OUTPUTDIR%\pandoc.pdf.log"
if %ERRORLEVEL% neq 0 (
    echo Error generating PDF. Check %OUTPUTDIR%\pandoc.pdf.log for details.
    exit /b 1
)
goto :eof

:all
call :pdf
goto :eof

endlocal