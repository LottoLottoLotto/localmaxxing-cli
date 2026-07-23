//go:build !windows && !linux

package main

import (
	"fmt"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"syscall"
)

func readProcessIdentity(pid int) (string, bool, error) {
	if pid <= 0 {
		return "", false, os.ErrProcessDone
	}
	if err := syscall.Kill(pid, 0); err != nil {
		return "", false, err
	}
	out, err := exec.Command("ps", "-p", strconv.Itoa(pid), "-o", "lstart=", "-o", "command=").Output()
	if err != nil {
		return "", false, err
	}
	identity := strings.TrimSpace(string(out))
	if identity == "" {
		return "", false, fmt.Errorf("ps returned no process identity for pid %d", pid)
	}
	return "unix:" + identity, true, nil
}
