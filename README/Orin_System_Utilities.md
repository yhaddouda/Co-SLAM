# Orin System Utilities

## Set Clock

### Turn off auto-sync:

```bash
sudo timedatectl set-ntp false

```

### Set the time manually:

```bash
sudo timedatectl set-time "2026-03-19 11:41:00"
```

## Change Power Mode


### Check Your Current Mode
Before changing anything, see what mode you are currently in:

```Bash
sudo nvpmodel -q
```
### Change the Power Mode
To switch modes, use the -m flag followed by the ID number of the mode:

```Bash
sudo nvpmodel -m <mode_id>
```

Mode IDs for AGX Orin (depends on the jetpack):
| ID | Mode Name 
| 0 | MAXN 
| 1 | 15W 
| 3 | 30W 
| 5 | 50W 

## Change Ownership Of A Folder From `sudo` To User

``` Bash
sudo chown -R user:user /path/to/folder
```

## For every nvpmodel set maximum performance
sudo jetson_clocks
