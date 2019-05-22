package main

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"syscall"

	xlog "github.com/qiniu/x/xlog.v7"
)

func runCmd(ctx context.Context, name string, dir, cmd string, args []string, env []string) {
	var xl = xlog.New(fmt.Sprintf("PM [%v]", name))
	c := exec.CommandContext(ctx, cmd, args...)
	c.Env = env
	c.Stderr = os.Stderr
	c.Stdout = os.Stdout
	c.Dir = dir
	c.SysProcAttr = &syscall.SysProcAttr{Setpgid: true, Pgid: os.Getpid()}
	xl.Info("run ", "path", c.Path, "args", c.Args, "env", c.Env)
	err := c.Run()
	if err != nil {
		xl.Error("run err", err)
		return
	}
	xl.Info("cmd exited")
}
