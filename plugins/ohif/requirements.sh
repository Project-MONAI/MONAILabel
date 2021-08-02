#!/bin/bash

if which yarn >/dev/null; then
  echo "node/yarn is already installed"
else
  sudo apt install npm -y
  sudo npm install --global yarn
fi
