Important, remember that to run this you need to set up the following commands in your terminal first:


```
SITE_PACKAGES=$(python3 -c "import sysconfig; print(sysconfig.get_paths()['purelib'])" 2>/dev/null)
export LD_LIBRARY_PATH="$SITE_PACKAGES/nvidia/cublas/lib:$SITE_PACKAGES/nvidia/cudnn/lib:$LD_LIBRARY_PATH"
```

```
python live_transcribe_sd.py --device 0
```
