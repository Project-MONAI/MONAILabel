#!/bin/bash

if which yarn >/dev/null; then
  echo "node/yarn is already installed"
else
  apt update -y
  apt install npm -y
  npm install --global yarn
fi
