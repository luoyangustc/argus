package main

import (
	"flag"
	"fmt"
	"go/build"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/fsnotify/fsnotify"
	"github.com/qiniu/log.v1"
)

func ce(err error) {
	if err != nil {
		log.Panicln(err)
	}
}

func findImport(p string) []string {
	pkgs := make(map[string]bool)
	var find func(p string)
	find = func(p string) {
		pkg, err := build.Import(p, ".", 0)
		if err != nil {
			log.Debug(err)
			return
		}
		for _, s := range pkg.Imports {
			if strings.HasPrefix(s, "qiniu.com") {
				find(s)
				pkgs[s] = true
			}
		}
	}
	find(p)
	ps := make([]string, 0)
	for p := range pkgs {
		ps = append(ps, p)
	}
	return ps
}

type app struct {
	pkg string
	dep []string
}

func reloadApp(pkg string, goremanPort int) {
	apps := pkgToApp(pkg)
	start := time.Now()
	buf, err := exec.Command("sh", "-c", "go install "+pkg).CombinedOutput()
	if err != nil {
		log.Warn("exec go install error", err, string(buf))
		return
	}
	log.Debug(string(buf))
	endBuild := time.Now()
	for _, app := range apps {
		buf, err = exec.Command("sh", "-c", fmt.Sprintf("./hack/run/bin/goreman -p %v run restart %v", goremanPort, app)).CombinedOutput()
		if err != nil {
			log.Warn("exec goreman error", err, string(buf))
			return
		}
	}
	log.Debug(string(buf))
	log.Info("reload", apps, "success", "build use time", endBuild.Sub(start), "restart use time", time.Since(endBuild))
}

func main() {
	var goremanPort int
	var dir string
	var err error
	{
		flag.IntVar(&goremanPort, "p", 0, "goreman port")
		flag.Parse()

		ce(os.Chdir(".."))
		_, err := os.Stat(".git")
		ce(err)
		dir, err = os.Getwd()
		ce(err)
	}

	var apps []app
	{
		err = filepath.Walk("cmd", func(path string, info os.FileInfo, err error) error {
			ce(err)
			if info.IsDir() && strings.HasPrefix(path, "cmd/") {
				pkg := "qiniu.com/argus/" + path
				dep := findImport(pkg)
				if len(dep) > 0 {
					apps = append(apps, app{pkg: pkg, dep: dep})
				}
			}
			return nil
		})
		ce(err)
		for _, app := range apps {
			log.Debug("\tdep ", app.pkg, app.dep)
		}
	}

	reverseDep := make(map[string][]string)
	{
		for _, app := range apps {
			for _, p := range app.dep {
				reverseDep[p] = append(reverseDep[p], app.pkg)
			}
		}
		log.Debug("----")
		for r, d := range reverseDep {
			log.Debug("\trdep ", r, d)
		}
	}

	log.Infof("start watch %q, app num %v, goreman port %v\n", dir, len(apps), goremanPort)
	watcher, err := NewRecursiveWatcher(dir)
	ce(err)
	defer watcher.Close()

	done := make(chan bool)
	go func() {
		for {
			select {
			case event := <-watcher.Events:
				log.Debug("event:", event)
				if event.Op&fsnotify.Write == fsnotify.Write {
					if n := strings.LastIndex(event.Name, "qiniu.com/argus/hack"); n != -1 {
						continue
					}
					log.Info("modified file:", event.Name)
					if n := strings.LastIndex(event.Name, "qiniu.com"); n != -1 {
						pkg := filepath.Dir(event.Name[n:])
						if r, ok := reverseDep[pkg]; ok {
							log.Info("changed pkg", pkg, "should reload app", r)
							time.Sleep(time.Second / 10)
							for _, v := range r {
								reloadApp(v, goremanPort)
							}
						} else {
							time.Sleep(time.Second / 10)
							if strings.HasPrefix(pkg, "qiniu.com/argus/cmd/") {
								log.Info("changed pkg", pkg, "should reload app", pkg)
								reloadApp(pkg, goremanPort)
							}
						}
					}
				}
			case err := <-watcher.Errors:
				log.Warn("error:", err)
			}
		}
	}()

	<-done
}

func pkgToApp(pkg string) (app []string) {
	if pkg == "qiniu.com/argus/cmd/serving-eval" {
		return []string{"hello-eval"}
	}
	return []string{filepath.Base(pkg)}
}
