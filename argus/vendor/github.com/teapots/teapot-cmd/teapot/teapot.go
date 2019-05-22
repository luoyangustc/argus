package main

import (
	"flag"
	"fmt"
	nlog "log"
	"os"
	"path/filepath"
	"strings"

	"github.com/teapots/teapot"
)

var (
	commands = map[string]*Command{}
	log      = func() teapot.Logger {
		l := teapot.NewLogger(nlog.New(os.Stderr, "", nlog.LstdFlags|nlog.Lmicroseconds))
		l.SetColorMode(true)
		l.SetLineInfo(false)
		l.SetFlatLine(false)
		return l
	}()

	Options     = new(options)
	optionsFlag = flag.NewFlagSet("options", flag.ExitOnError)
)

func init() {
	wd, _ := os.Getwd()
	appName := filepath.Base(wd)
	optionsFlag.StringVar(&Options.AppName, "app", appName, "app name")
}

type options struct {
	AppName string
}

type Command struct {
	Flag *flag.FlagSet
	Cmd  CommandRunner
}

type CommandRunner interface {
	Run()
}

func printHelp() {
	fmt.Fprintln(os.Stderr, `Usage: teapot COMMAND [arg...]

Teapot helper tool.

Options:
    -app    app name

Commands:
    run     auto watch and rebuild project

Run 'teapot COMMAND --help' for more information on a command.`)
	os.Exit(2)
}

func main() {
	var args = os.Args[1:]
	var optArgs []string
	var cmd string

	if len(args) > 0 && args[0] == "--help" {
		printHelp()
	}

	for i := 0; i < len(args); i++ {
		if !strings.HasPrefix(args[0], "-") {
			break
		}
		optArgs = append(optArgs, args[i])
		args = args[1:]
	}

	if len(optArgs) > 0 {
		optionsFlag.Parse(optArgs)
	}

	if len(args) > 0 {
		cmd = args[0]
	}

	if commands[cmd] == nil {
		printHelp()
	}

	log.Infof("use '%s' as app name", red(Options.AppName))

	commands[cmd].Flag.Parse(args[1:])
	commands[cmd].Cmd.Run()
}
