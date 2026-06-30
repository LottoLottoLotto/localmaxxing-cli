package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestSkillEmbedContainsSkillAndReferences(t *testing.T) {
	data, err := skillFS.ReadFile("skill/SKILL.md")
	if err != nil {
		t.Fatalf("embedded SKILL.md unreadable: %v", err)
	}
	text := string(data)
	if !strings.Contains(text, "name: localmaxxing-cli") {
		t.Fatal("embedded SKILL.md is missing localmaxxing-cli name")
	}
	if !strings.Contains(text, "description:") {
		t.Fatal("embedded SKILL.md is missing description")
	}

	for _, path := range []string{
		"skill/references/benchmarks.md",
		"skill/references/evals.md",
		"skill/references/hardware-and-setups.md",
		"skill/references/reference.md",
	} {
		ref, err := skillFS.ReadFile(path)
		if err != nil {
			t.Fatalf("embedded reference %s unreadable: %v", path, err)
		}
		if len(ref) == 0 {
			t.Fatalf("embedded reference %s is empty", path)
		}
	}

	onDisk, err := os.ReadFile(filepath.Join("skill", "SKILL.md"))
	if err != nil {
		t.Fatalf("on-disk SKILL.md unreadable: %v", err)
	}
	if !bytes.Equal(data, onDisk) {
		t.Fatal("embedded SKILL.md differs from on-disk skill/SKILL.md")
	}
}

func TestSkillInstallWritesTree(t *testing.T) {
	dir := t.TempDir()
	if err := runWithArgs(parseArgs([]string{"skill", "install", "--dir", dir, "--quiet"})); err != nil {
		t.Fatalf("skill install returned error: %v", err)
	}

	installedSkill := filepath.Join(dir, skillName, "SKILL.md")
	installedRef := filepath.Join(dir, skillName, "references", "benchmarks.md")
	data, err := os.ReadFile(installedSkill)
	if err != nil {
		t.Fatalf("installed SKILL.md unreadable: %v", err)
	}
	if _, err := os.ReadFile(installedRef); err != nil {
		t.Fatalf("installed benchmarks.md unreadable: %v", err)
	}
	embedded, err := skillFS.ReadFile("skill/SKILL.md")
	if err != nil {
		t.Fatalf("embedded SKILL.md unreadable: %v", err)
	}
	if !bytes.Equal(data, embedded) {
		t.Fatal("installed SKILL.md differs from embedded SKILL.md")
	}
}

func TestSkillPrintWritesToOut(t *testing.T) {
	out := filepath.Join(t.TempDir(), "SKILL.md")
	if err := runWithArgs(parseArgs([]string{"skill", "print", "--out", out, "--quiet"})); err != nil {
		t.Fatalf("skill print returned error: %v", err)
	}
	data, err := os.ReadFile(out)
	if err != nil {
		t.Fatalf("printed SKILL.md unreadable: %v", err)
	}
	embedded, err := skillFS.ReadFile("skill/SKILL.md")
	if err != nil {
		t.Fatalf("embedded SKILL.md unreadable: %v", err)
	}
	if !bytes.Equal(data, embedded) {
		t.Fatal("printed SKILL.md differs from embedded SKILL.md")
	}
}

func TestSkillUnknownSubcommandErrors(t *testing.T) {
	requireCliErrorCode(t, runWithArgs(parseArgs([]string{"skill", "bogus"})), "unknown_skill_command")
}
