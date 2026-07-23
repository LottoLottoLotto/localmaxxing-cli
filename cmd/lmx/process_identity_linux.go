//go:build linux

package main

import (
	"fmt"
	"os"
	"strings"
)

func readProcessIdentity(pid int) (string, bool, error) {
	if pid <= 0 {
		return "", false, os.ErrProcessDone
	}
	stat, err := os.ReadFile(fmt.Sprintf("/proc/%d/stat", pid))
	if err != nil {
		return "", false, err
	}
	statText := string(stat)
	commEnd := strings.LastIndexByte(statText, ')')
	if commEnd < 0 {
		return "", false, fmt.Errorf("invalid process stat for pid %d", pid)
	}
	fields := strings.Fields(statText[commEnd+1:])
	if len(fields) <= 19 {
		return "", false, fmt.Errorf("incomplete process stat for pid %d", pid)
	}
	bootID, err := os.ReadFile("/proc/sys/kernel/random/boot_id")
	if err != nil {
		return "", false, err
	}
	boot := strings.TrimSpace(string(bootID))
	if boot == "" || fields[19] == "" {
		return "", false, fmt.Errorf("incomplete process identity for pid %d", pid)
	}
	return "linux:" + boot + ":" + fields[19], fields[0] != "Z", nil
}
