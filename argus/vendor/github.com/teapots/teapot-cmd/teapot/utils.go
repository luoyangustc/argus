package main

import (
	"bytes"
	"crypto/md5"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

type sliceValue []string

func (s *sliceValue) Set(v string) error {
	for _, p := range strings.Split(v, ",") {
		p = strings.TrimSpace(p)
		if p != "" {
			*s = append(*s, v)
		}
	}
	return nil
}

func (s *sliceValue) String() string {
	return strings.Join(*s, ",")
}

func (s *sliceValue) Unique() {
	v := make(sliceValue, 0, len(*s))
	m := make(map[string]bool, len(*s))
	for _, p := range *s {
		if m[p] {
			continue
		}
		m[p] = true
		v = append(v, p)
	}
	*s = v
}

func isFile(path string) bool {
	f, err := os.OpenFile(path, os.O_RDONLY, 0644)
	if err != nil {
		return false
	}
	stat, err := f.Stat()
	if err != nil {
		return false
	}
	if stat.IsDir() {
		return false
	}
	return true
}

func isDir(path string) bool {
	f, err := os.OpenFile(path, os.O_RDONLY, 0644)
	if err != nil {
		return false
	}
	stat, err := f.Stat()
	if err != nil {
		return false
	}
	if stat.IsDir() {
		return true
	}
	return false
}

func readFile(path string) string {
	body, _ := ioutil.ReadFile(path)
	return string(body)
}

func hashFile(path string) string {
	m := md5.New()
	m.Write([]byte(readFile(path)))
	return string(m.Sum(nil))
}

func fileModUnix(path string) int64 {
	stat, err := os.Stat(path)
	if err != nil {
		return 0
	}
	return stat.ModTime().Unix()
}

func decodeJson(p interface{}, body string) {
	json.Unmarshal([]byte(body), p)
}

func parsePaths(paths sliceValue) sliceValue {
	var res sliceValue
	for _, p := range paths {
		switch {
		case strings.HasPrefix(p, "$"):
			p = strings.TrimPrefix(p, "$")
			val := os.Getenv(p)
			list := filepath.SplitList(val)
			res = append(res, parsePaths(list)...)
		case p == ".":
			dir, _ := os.Getwd()
			dir, _ = filepath.Abs(dir)
			res = append(res, dir)
		default:
			p, _ = filepath.Abs(p)
			res = append(res, p)
		}
	}
	return res
}

func logCmdError(cmd *exec.Cmd, buf *bytes.Buffer, extras ...interface{}) {
	msg := strings.Join(cmd.Args, " ") + "\n" + strings.TrimSpace(buf.String())
	if len(extras) > 0 {
		msg += "\n" + fmt.Sprint(extras...)
	}
	log.Error(msg)
}

func goList(path string, envs []string) (*exec.Cmd, *bytes.Buffer, error) {
	cmd := exec.Command("go", "list", path)
	buf := bytes.NewBufferString("")
	cmd.Stdout = buf
	cmd.Stderr = buf
	cmd.Env = append(os.Environ(), runConfig.Envs...)
	err := cmd.Run()
	return cmd, buf, err
}

func goInstall(dir, pkg string, envs []string) (*exec.Cmd, *bytes.Buffer, error) {
	buf := bytes.NewBufferString("")
	args := []string{"install"}
	if pkg != "" {
		args = append(args, dir)
	}
	cmd := exec.Command("go", args...)
	cmd.Dir = dir
	cmd.Stdout = buf
	cmd.Stderr = buf
	cmd.Env = append(os.Environ(), envs...)
	err := cmd.Run()

	switch {
	case strings.Contains(buf.String(), "no buildable Go source files"):
		err = nil
	}
	return cmd, buf, err
}

func red(con string) string {
	return "\033[1;31m" + con + "\033[0m"
}
