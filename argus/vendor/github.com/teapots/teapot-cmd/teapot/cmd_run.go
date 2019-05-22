package main

import (
	"bytes"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/howeyc/fsnotify"
)

type runJson struct {
	Exts sliceValue `json:"exts"`
	Dirs sliceValue `json:"dirs"`
	Args sliceValue `json:"args"`
	Envs sliceValue `json:"envs"`
	Out  string     `json:"out"`

	MainFiles sliceValue `json:"main_files"`
}

var (
	runConfigFile string
	runConfig     = new(runJson)
)

func init() {
	f := flag.NewFlagSet("run", flag.ExitOnError)
	runConfig.Dirs = sliceValue{".", "$GOPATH"}
	runConfig.Exts = sliceValue{".go"}

	f.StringVar(&runConfigFile, "conf", "run.json", "config json file")
	f.Var(&runConfig.Exts, "ext", "watch file ext")
	f.Var(&runConfig.Dirs, "dir", "watch directories")
	f.Var(&runConfig.Args, "arg", "program run args")
	f.Var(&runConfig.Envs, "env", "program run envs")
	f.StringVar(&runConfig.Out, "out", "", "output binary path")
	f.Var(&runConfig.MainFiles, "main", "main go files")

	commands["run"] = &Command{Flag: f, Cmd: &commandRun{}}
}

type commandRun struct {
	sync.Mutex

	appName string

	signal chan os.Signal

	running *exec.Cmd
	period  time.Time
}

func (c *commandRun) Run() {
	if isFile(runConfigFile) {
		log.Infof("detected run config file '%s'", red(runConfigFile))
		decodeJson(&runConfig, readFile(runConfigFile))
	}

	path, _ := os.Getwd()
	os.Chdir(path)

	c.appName = Options.AppName
	if runtime.GOOS == "windows" {
		c.appName += ".exe"
	}

	if runConfig.Out == "" {
		runConfig.Out, _ = filepath.Abs(filepath.Join("./", c.appName))
	}

	c.signal = make(chan os.Signal)

	signal.Notify(c.signal,
		syscall.SIGINT,
		syscall.SIGTERM,
		syscall.SIGQUIT,
	)

	c.parseRunConfig()
	c.startWatch()

	// after first build
	goInstall("", "./...", []string(runConfig.Envs))

	go c.buildAndRun()

	c.wait()
}

func (c *commandRun) wait() {
	for {
		select {
		case sig := <-c.signal:
			switch sig {
			case syscall.SIGINT, syscall.SIGTERM, syscall.SIGQUIT:
				c.kill()
				log.Info("bye bye !")
				os.Exit(0)
			}
		}
	}
}

func (c *commandRun) parseRunConfig() {
	// default use .go ext
	runConfig.Exts.Set(".go")

	// parse dir config
	if len(runConfig.Dirs) > 2 {
		runConfig.Dirs = runConfig.Dirs[2:]
	}

	runConfig.Dirs = parsePaths(runConfig.Dirs)

	// unqiue config
	runConfig.Exts.Unique()
	runConfig.Dirs.Unique()
	runConfig.MainFiles.Unique()
}

func (c *commandRun) startWatch() {
	watcher, err := fsnotify.NewWatcher()
	if err != nil {
		log.Errorf("failed start watcher ''", err)
		return
	}

	exts := make(map[string]bool, len(runConfig.Exts))
	for _, ext := range runConfig.Exts {
		exts[ext] = true
	}

	var (
		modTimes = make(map[string]int64)
		watches  = make(map[string]bool)
	)
	go func() {
		for {
			select {
			case evt := <-watcher.Event:
				path := evt.Name
				// log.Info(path)

				// skip private
				name := filepath.Base(path)
				if strings.HasPrefix(name, ".") {
					continue
				}

				switch {
				case evt.IsDelete(), evt.IsRename():
					// when path exist in watches
					// remove watch and rebuild
					if watches[path] {
						delete(watches, path)
						watcher.RemoveWatch(path)
					}
				case evt.IsCreate() && isDir(path):
					// if dir not exist in waxxtches
					// add it to wathces
					if !watches[path] {
						watches[path] = true
						watcher.Watch(path)
					}
				}

				// skip non-exist file
				if !exts[filepath.Ext(path)] {
					continue
				}

				// detect file mod time
				newUnix := fileModUnix(path)
				if modTimes[path] == newUnix {
					continue
				}
				if evt.IsDelete() || evt.IsRename() {
					delete(modTimes, path)
				} else {
					modTimes[path] = newUnix
				}

				log.Info(red("detected"), evt.String())

				go func() {
					pkg := filepath.Dir(path)
					goInstall(pkg, "", []string(runConfig.Envs))

					c.buildAndRun()
				}()

			case err := <-watcher.Error:
				log.Errorf("watch error: %v", err)
			}
		}
	}()

	log.Info(red("watching..."))

	for _, dir := range runConfig.Dirs {
		log.Infof("watch dir: %s", dir)
		filepath.Walk(dir, func(path string, info os.FileInfo, err error) (ret error) {
			if err != nil {
				log.Error("walk dir:", err)
				return
			}

			// skip private
			if info.IsDir() && strings.HasPrefix(info.Name(), ".") {
				ret = filepath.SkipDir
			}

			if info.IsDir() {
				return
			}

			dir, file := filepath.Split(path)
			if !exts[filepath.Ext(file)] {
				return
			}

			dir = filepath.Clean(dir)
			if watches[dir] {
				return
			}

			watches[dir] = true
			watcher.Watch(dir)
			return
		})
	}
}

func (c *commandRun) buildAndRun() (done bool) {
	point := time.Now()
	c.period = point

	c.Lock()

	time.Sleep(200 * time.Millisecond)

	defer func() {
		if done {
			c.period = time.Now()
		}
		c.Unlock()
	}()

	if !point.Equal(c.period) {
		return
	}

	log.Info(red("building... " + strings.Join(runConfig.MainFiles, " ")))

	args := append([]string{"build", "-o", runConfig.Out}, runConfig.MainFiles...)
	cmd := exec.Command("go", args...)
	buf := bytes.NewBufferString("")
	cmd.Stdout = buf
	cmd.Stderr = buf
	err := cmd.Run()
	if err != nil {
		logCmdError(cmd, buf, err)
		return
	}

	log.Info(red("done..."))

	c.kill()

	// wait app exit
	var wait time.Duration = 1
	for c.running != nil {
		if c.running.Process == nil {
			break
		}
		cmd := exec.Command("kill", "-0", fmt.Sprint(c.running.Process.Pid))
		if cmd.Run() != nil {
			c.running = nil
			break
		}
		time.Sleep(wait * time.Millisecond)
		if wait < 1000 {
			wait += wait
		} else {
			wait = 1000
		}
	}

	log.Infof(red("starting")+" %s", runConfig.Out)

	// run app
	c.running = exec.Command(runConfig.Out)
	c.running.Stdin = os.Stdin
	c.running.Stdout = os.Stdout
	c.running.Stderr = os.Stderr
	c.running.Args = append([]string{runConfig.Out}, runConfig.Args...)
	c.running.Env = append(os.Environ(), runConfig.Envs...)

	go func() {
		err := c.running.Run()
		if err != nil {
			log.Error(err)
			return
		}
		log.Infof("%s is running", c.appName)
	}()

	done = true
	return
}

func (c *commandRun) kill() {
	// if process exist try to kill it
	if c.running != nil {
		if c.running.Process != nil {
			c.running.Process.Kill()
		}
	}
}
