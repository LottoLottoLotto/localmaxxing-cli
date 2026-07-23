//go:build windows

package main

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"os/signal"
	"strconv"
	"syscall"
	"unsafe"
)

const (
	lockFileExclusiveLock    = 0x00000002
	lockFileFailImmediately  = 0x00000001
	processQueryLimitedInfo  = 0x00001000
	stillActive              = 259
	processIdentityPrefix    = "windows-filetime:"
	terminalRunLockByteCount = 1
	windowsSigBreak          = syscall.Signal(21)
)

var (
	kernel32DLL                  = syscall.NewLazyDLL("kernel32.dll")
	procGenerateConsoleCtrlEvent = kernel32DLL.NewProc("GenerateConsoleCtrlEvent")
	procLockFileEx               = kernel32DLL.NewProc("LockFileEx")
	procUnlockFileEx             = kernel32DLL.NewProc("UnlockFileEx")
)

func configureCommandProcessGroup(cmd *exec.Cmd) {}

func configureDetachedCommand(cmd *exec.Cmd) {
	if cmd.SysProcAttr == nil {
		cmd.SysProcAttr = &syscall.SysProcAttr{}
	}
	cmd.SysProcAttr.CreationFlags |= syscall.CREATE_NEW_PROCESS_GROUP
}

func killCommandProcessGroup(cmd *exec.Cmd) {
	if cmd == nil || cmd.Process == nil {
		return
	}
	_ = exec.Command("taskkill", "/T", "/F", "/PID", fmt.Sprint(cmd.Process.Pid)).Run()
}

func processIsRunning(pid int) bool {
	_, err := captureProcessIdentity(pid)
	return err == nil
}

func captureProcessIdentity(pid int) (string, error) {
	if !validWindowsPID(pid) {
		return "", os.ErrProcessDone
	}
	handle, err := syscall.OpenProcess(processQueryLimitedInfo, false, uint32(pid))
	if err != nil {
		return "", fmt.Errorf("OpenProcess(%d): %w", pid, err)
	}
	defer syscall.CloseHandle(handle)

	var creationTime, exitTime, kernelTime, userTime syscall.Filetime
	if err := syscall.GetProcessTimes(handle, &creationTime, &exitTime, &kernelTime, &userTime); err != nil {
		return "", fmt.Errorf("GetProcessTimes(%d): %w", pid, err)
	}
	var exitCode uint32
	if err := syscall.GetExitCodeProcess(handle, &exitCode); err != nil {
		return "", fmt.Errorf("GetExitCodeProcess(%d): %w", pid, err)
	}
	if exitCode != stillActive {
		return "", os.ErrProcessDone
	}
	creationTicks := uint64(creationTime.HighDateTime)<<32 | uint64(creationTime.LowDateTime)
	return processIdentityPrefix + strconv.FormatUint(creationTicks, 16), nil
}

func processMatchesIdentity(pid int, identity string) bool {
	if identity == "" {
		return false
	}
	current, err := captureProcessIdentity(pid)
	return err == nil && current == identity
}

func signalDetachedProcess(pid int, force bool) error {
	if !validWindowsPID(pid) {
		return os.ErrProcessDone
	}
	if force {
		return exec.Command("taskkill", "/T", "/F", "/PID", strconv.Itoa(pid)).Run()
	}
	r1, _, callErr := procGenerateConsoleCtrlEvent.Call(
		uintptr(syscall.CTRL_BREAK_EVENT),
		uintptr(uint32(pid)),
	)
	if r1 == 0 {
		return windowsCallError("GenerateConsoleCtrlEvent", callErr)
	}
	return nil
}

func terminalSignalContext() (context.Context, context.CancelFunc) {
	return signal.NotifyContext(context.Background(), os.Interrupt, windowsSigBreak)
}

func lockTerminalRunFile(file *os.File) error {
	if file == nil {
		return os.ErrInvalid
	}
	var overlapped syscall.Overlapped
	r1, _, callErr := procLockFileEx.Call(
		file.Fd(),
		lockFileExclusiveLock|lockFileFailImmediately,
		0,
		terminalRunLockByteCount,
		0,
		uintptr(unsafe.Pointer(&overlapped)),
	)
	if r1 == 0 {
		return windowsCallError("LockFileEx", callErr)
	}
	return nil
}

func unlockTerminalRunFile(file *os.File) error {
	if file == nil {
		return os.ErrInvalid
	}
	var overlapped syscall.Overlapped
	r1, _, callErr := procUnlockFileEx.Call(
		file.Fd(),
		0,
		terminalRunLockByteCount,
		0,
		uintptr(unsafe.Pointer(&overlapped)),
	)
	if r1 == 0 {
		return windowsCallError("UnlockFileEx", callErr)
	}
	return nil
}

func validWindowsPID(pid int) bool {
	return pid > 0 && uint64(pid) <= uint64(^uint32(0))
}

func windowsCallError(api string, callErr error) error {
	if callErr == nil || callErr == syscall.Errno(0) {
		callErr = syscall.EINVAL
	}
	return fmt.Errorf("%s: %w", api, callErr)
}
