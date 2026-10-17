#!/bin/bash
# Log in to Delta once (password + Duo) and keep a shared connection open for 8 hours.
# While this window stays open, other ssh/rsync to Delta from this computer reuse it: no new Duo push.
#   bash delta_login.sh
mkdir -p -m 700 ~/.ssh
read -r -p "Delta NetID: " NETID
exec ssh -o ControlMaster=auto -o "ControlPath=~/.ssh/cm-%r@%h:%p" -o ControlPersist=8h \
    "$NETID@login.delta.ncsa.illinois.edu"
