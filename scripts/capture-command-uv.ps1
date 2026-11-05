<#
.SYNOPSIS
    Captures the exact PowerShell command line used to invoke `uv` and passes
    it to child Python processes through LSTM_AUTOENCODER_COMMAND.

.USAGE
    Dot-source this file once in the current PowerShell session:
        . .\scripts\capture-command-uv.ps1

    Then invoke UV normally, for example:
        uv run python -m lstm_autoencoder train-model --dataset paysim_synthetic_financial --window 1 --bottleneck --seeds 42,123,2026

    The command above is captured before UV starts Python.
#>

function uv {
    [CmdletBinding()]
    param(
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]] $UvArguments
    )

    $uvApplication = Get-Command -Name 'uv.exe' -CommandType Application -ErrorAction Stop |
        Select-Object -First 1

    # In an interactive PowerShell session, Line contains the command as typed.
    # Keep a reconstructed fallback for calls made without an interactive line.
    $capturedCommand = $MyInvocation.Line
    if ([string]::IsNullOrWhiteSpace($capturedCommand)) {
        $capturedCommand = 'uv ' + ($UvArguments -join ' ')
    }

    $hadPreviousValue = Test-Path Env:LSTM_AUTOENCODER_COMMAND
    $previousValue = $env:LSTM_AUTOENCODER_COMMAND
    $exitCode = 0

    try {
        $env:LSTM_AUTOENCODER_COMMAND = $capturedCommand
        & $uvApplication.Source @UvArguments
        $exitCode = $LASTEXITCODE
    }
    finally {
        if ($hadPreviousValue) {
            $env:LSTM_AUTOENCODER_COMMAND = $previousValue
        }
        else {
            Remove-Item Env:LSTM_AUTOENCODER_COMMAND -ErrorAction SilentlyContinue
        }
    }

    $global:LASTEXITCODE = $exitCode
}
