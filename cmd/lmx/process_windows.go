//go:build windows

package main

import (
	"fmt"
	"os/exec"
)

func configureCommandProcessGroup(cmd *exec.Cmd) {}

func killCommandProcessGroup(cmd *exec.Cmd) {
	if cmd == nil || cmd.Process == nil {
		return
	}
	_ = exec.Command("taskkill", "/T", "/F", "/PID", fmt.Sprint(cmd.Process.Pid)).Run()
}
