ARTI 503 - Project Stage 1 - Sequential benchmark (group 9MA2G2)
=================================================================

You only need to run ONE thing and send ONE file back.

BEFORE YOU START
  - Plug in your charger.
  - Close other heavy programs (games, browsers with many tabs, video calls).
  - You need Python 3.8 or newer. If you do not have it:
    https://www.python.org/downloads/
    (Windows: during setup tick "Add python.exe to PATH")

STEP 1 - Unzip
  Right-click the zip -> "Extract All". Do not run it from inside the zip.

STEP 2 - Run
  Windows:      double-click  run_windows.bat
  Mac / Linux:  open Terminal in this folder and type
                    sh run_mac_linux.sh
  Or any system:    python run.py --name YourName

  It asks for your name (English letters), then runs by itself.
  It takes about 15-40 minutes depending on your laptop. While it runs:
    - do not close the window
    - do not let the laptop sleep (keep the lid open)
    - do not use the laptop for heavy work (games, video calls, compiling)
  If NumPy is missing it installs it automatically (needs internet once).

STEP 3 - Send
  When it says DONE, send this file to the group:
      results/RESULTS_<YourName>.json
  (results/RESULTS_<YourName>.txt is a readable copy for you.)

IF SOMETHING GOES WRONG
  Send the RESULTS_<YourName>.json file anyway. It contains whatever finished
  and the error details. Do not edit any of the files.

WHAT IT DOES
  Runs our sequential near-duplicate detector (MinHash + LSH) on 1,000,
  10,000, 50,000 and 100,000 real Arabic web documents (FineWeb-2, from
  Common Crawl), repeats each size 3-5 times, and records the time of every
  stage plus your CPU details. Everyone runs exactly the same data, so the
  checksums in the results must match across the team.
