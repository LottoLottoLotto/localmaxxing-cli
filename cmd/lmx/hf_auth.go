package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// Hugging Face credentials follow huggingface_hub's own lookup, so one
// `hf auth login` (or `lmx auth hf --token-stdin`) serves lmx, vLLM, SGLang and
// transformers alike: HF_TOKEN, then the token file at HF_TOKEN_PATH, else
// $HF_HOME/token, where HF_HOME defaults to $XDG_CACHE_HOME/huggingface or
// ~/.cache/huggingface.

func huggingFaceHome() string {
	if home := os.Getenv("HF_HOME"); home != "" {
		return home
	}
	base := os.Getenv("XDG_CACHE_HOME")
	if base == "" {
		userHome, err := os.UserHomeDir()
		if err != nil {
			return ""
		}
		base = filepath.Join(userHome, ".cache")
	}
	return filepath.Join(base, "huggingface")
}

func huggingFaceTokenPath() string {
	if path := os.Getenv("HF_TOKEN_PATH"); path != "" {
		return path
	}
	if home := huggingFaceHome(); home != "" {
		return filepath.Join(home, "token")
	}
	return ""
}

// huggingFaceToken returns the active token and where it came from ("HF_TOKEN"
// or the token file path), or empty strings when none is configured.
func huggingFaceToken() (string, string) {
	if token := strings.TrimSpace(os.Getenv("HF_TOKEN")); token != "" {
		return token, "HF_TOKEN"
	}
	path := huggingFaceTokenPath()
	if path == "" {
		return "", ""
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return "", ""
	}
	if token := strings.TrimSpace(string(data)); token != "" {
		return token, path
	}
	return "", ""
}

// withHuggingFaceToken gives a server child process the stored token when the
// environment lacks one. Managed engines use a private HF_HOME, and systemd
// deployments do not inherit the shell, so neither would find it otherwise.
// The value lives only in the child's environment, never in argv or records.
func withHuggingFaceToken(environment []string) []string {
	for _, entry := range environment {
		if strings.HasPrefix(entry, "HF_TOKEN=") {
			return environment
		}
	}
	token, _ := huggingFaceToken()
	if token == "" {
		return environment
	}
	return append(environment, "HF_TOKEN="+token)
}

// huggingFaceWhoami validates a token against the Hub and returns the account.
func huggingFaceWhoami(args cliArgs, token string) (map[string]any, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, strings.TrimRight(hfAPIURL(args), "/")+"/api/whoami-v2", nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+token)
	client := modelPullHTTPClient()
	defer client.CloseIdleConnections()
	response, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer response.Body.Close()
	if response.StatusCode == http.StatusUnauthorized {
		return nil, errors.New("Hugging Face rejected the token (401)")
	}
	if response.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("Hugging Face whoami returned HTTP %d", response.StatusCode)
	}
	var body map[string]any
	if err := decodeJSONBody(response, &body); err != nil {
		return nil, err
	}
	account := map[string]any{"account": body["name"]}
	if access := asObject(asObject(body["auth"])["accessToken"]); access != nil {
		account["tokenName"] = access["displayName"]
		account["role"] = access["role"]
	}
	return account, nil
}

func decodeJSONBody(response *http.Response, destination any) error {
	data, err := io.ReadAll(io.LimitReader(response.Body, 1<<20))
	if err != nil {
		return err
	}
	return json.Unmarshal(data, destination)
}

// writeHuggingFaceToken stores the token where huggingface_hub reads it,
// readable only by the user, replacing any previous token atomically.
func writeHuggingFaceToken(path, token string) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(filepath.Dir(path), ".token-*")
	if err != nil {
		return err
	}
	defer os.Remove(tmp.Name())
	if err := tmp.Chmod(0o600); err != nil {
		tmp.Close()
		return err
	}
	if _, err := tmp.WriteString(token); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Sync(); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmp.Name(), path)
}

// handleAuthHF: `lmx auth hf --token-stdin` saves, `status` reports, `logout` removes.
func handleAuthHF(action string, args cliArgs) error {
	path := huggingFaceTokenPath()
	switch action {
	case "logout":
		if path == "" {
			return errors.New("cannot locate the Hugging Face token file")
		}
		if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
			return err
		}
		fields := map[string]any{"path": path}
		if os.Getenv("HF_TOKEN") != "" {
			fields["warning"] = "HF_TOKEN is still set in this environment and stays active until you unset it."
		}
		printInfo(args, "hf_auth_cleared", fields)
		return nil
	case "", "status":
		if action == "" && hasFlag(args, "token-stdin") {
			break
		}
		token, source := huggingFaceToken()
		if token == "" {
			printInfo(args, "hf_auth_missing", map[string]any{"path": path, "next": "Run lmx auth hf --token-stdin (paste a token from https://huggingface.co/settings/tokens), or hf auth login."})
			return nil
		}
		fields := map[string]any{"source": source, "token": redactKey(token)}
		if info, err := os.Stat(source); err == nil && info.Mode().Perm()&0o077 != 0 {
			fields["warning"] = fmt.Sprintf("%s is readable by other users (mode %o); run chmod 600 on it", source, info.Mode().Perm())
		}
		if account, err := huggingFaceWhoami(args, token); err != nil {
			fields["valid"] = false
			fields["error"] = err.Error()
		} else {
			fields["valid"] = true
			for key, value := range account {
				fields[key] = value
			}
		}
		printInfo(args, "hf_auth_status", fields)
		return nil
	default:
		return cliError{"unknown_command", "Unknown auth hf action " + action + ".", []string{"Use lmx auth hf --token-stdin, lmx auth hf status, or lmx auth hf logout."}, nil}
	}
	if path == "" {
		return errors.New("cannot locate the Hugging Face token file")
	}
	token, err := readKeyFromStdin(os.Stdin)
	if err != nil {
		return cliError{"missing_hf_token", "No Hugging Face token was provided on stdin.", []string{"Pipe a token to lmx auth hf --token-stdin."}, nil}
	}
	account, err := huggingFaceWhoami(args, token)
	if err != nil {
		return cliError{"hf_token_invalid", "The token was not saved: " + err.Error(), []string{"Create a read token at https://huggingface.co/settings/tokens."}, nil}
	}
	if err := writeHuggingFaceToken(path, token); err != nil {
		return err
	}
	fields := map[string]any{"path": path, "token": redactKey(token)}
	for key, value := range account {
		fields[key] = value
	}
	if os.Getenv("HF_TOKEN") != "" {
		fields["warning"] = "HF_TOKEN is set in this environment and takes precedence over the saved token."
	}
	printInfo(args, "hf_auth_saved", fields)
	return nil
}
