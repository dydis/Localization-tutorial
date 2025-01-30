@echo off
setlocal enabledelayedexpansion

:: Check if Pandoc is installed
where pandoc >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Error: Pandoc is not installed or not in PATH
    echo Please install Pandoc from https://pandoc.org/installing.html
    exit /b 1
)

:: Set directories
set "input_dir=markdown-documents"
set "output_dir=pdf_output"
set "listings_file=resources\tex\listings-2.tex"

:: Check if input directory exists
if not exist "%input_dir%" (
    echo Error: Input directory does not exist: %input_dir%
    echo Please create a 'markdown-documents' folder and place your markdown files there
    exit /b 1
)

:: Check if listings setup file exists
if not exist "%listings_file%" (
    echo Error: listings-setup.tex file not found at: %listings_file%
    echo Please ensure the file exists in the resources\tex directory
    exit /b 1
)

:: Create output directory if it doesn't exist
if not exist "%output_dir%" mkdir "%output_dir%"

:: Counter for processed files
set "processed=0"
set "failed=0"

:: Process each markdown file in the markdown-documents directory
for %%F in ("%input_dir%\*.md") do (
    echo Processing: %%F
    
    :: Create output filename
    set "input_file=%%F"
    set "output_file=%output_dir%\%%~nF.pdf"
    
    :: Convert markdown to PDF using pandoc with listings
    pandoc "%%F" -o "!output_file!" ^
        --pdf-engine=xelatex ^
        -V geometry:margin=1in ^
        -V papersize=a4 ^
        --standalone ^
        --toc ^
        --listings ^
        -H "%listings_file%"
        
    if !ERRORLEVEL! equ 0 (
        echo Success: Created !output_file!
        set /a "processed+=1"
    ) else (
        echo Failed to convert %%F
        set /a "failed+=1"
    )
)

:: Display summary
echo.
echo Conversion Complete
echo -----------------
echo Input directory: %input_dir%
echo Processed: %processed% files
echo Failed: %failed% files
echo Output directory: %output_dir%

if %failed% gtr 0 (
    echo Some conversions failed. Check error messages above.
    exit /b 1
)

endlocal