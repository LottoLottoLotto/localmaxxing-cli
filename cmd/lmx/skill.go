package main

import (
	"embed"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
)

//go:embed all:skill
var skillFS embed.FS

const skillName = "localmaxxing-cli"

// handleSkill implements `lmx skill print|install`.
func handleSkill(sub string, args cliArgs) error {
	switch sub {
	case "", "print":
		return printSkill(args)
	case "install":
		return installSkill(args)
	default:
		return cliError{"unknown_skill_command", fmt.Sprintf("Unknown skill subcommand %q.", sub),
			[]string{"Use `lmx skill print` to print SKILL.md, or `lmx skill install --dir <dir>` to write the skill files."}, nil}
	}
}

func printSkill(args cliArgs) error {
	data, err := skillFS.ReadFile("skill/SKILL.md")
	if err != nil {
		return err
	}
	if out := opt(args, "out"); out != "" {
		if err := os.WriteFile(out, data, 0o644); err != nil {
			return err
		}
		printStatus(args, "skill_written", map[string]any{"path": out, "bytes": len(data)})
		if humanOutput(args) {
			fmt.Printf("Wrote %s SKILL.md to %s\n", skillName, out)
		}
		return nil
	}
	if hasFlag(args, "json") {
		printJSON(args, map[string]any{"name": skillName, "content": string(data)})
	} else {
		fmt.Print(string(data))
	}
	return nil
}

func installSkill(args cliArgs) error {
	dir := firstNonEmpty(opt(args, "dir"), filepath.Join(".claude", "skills"))
	root := filepath.Join(dir, skillName)
	count := 0
	err := fs.WalkDir(skillFS, "skill", func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			return nil
		}
		rel := strings.TrimPrefix(p, "skill/")
		target := filepath.Join(root, rel)
		if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
			return err
		}
		data, err := skillFS.ReadFile(p)
		if err != nil {
			return err
		}
		if err := os.WriteFile(target, data, 0o644); err != nil {
			return err
		}
		count++
		return nil
	})
	if err != nil {
		return err
	}
	printStatus(args, "skill_installed", map[string]any{"dir": root, "files": count})
	if humanOutput(args) {
		fmt.Printf("Installed %s skill (%d files) to %s\n", skillName, count, root)
		fmt.Printf("Point your agent's skills directory at %s, or pass --dir to target ~/.claude/skills, .github/skills, etc.\n", filepath.Dir(root))
	}
	return nil
}
