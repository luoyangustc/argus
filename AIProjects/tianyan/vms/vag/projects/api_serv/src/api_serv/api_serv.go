package main

import (
	"fmt"
	"httpserv"
	"os"
	"runtime"

	"qiniupkg.com/x/config.v7"
	"qiniupkg.com/x/log.v7"
)

func main() {

	runtime.GOMAXPROCS(runtime.NumCPU())

	var conf httpserv.Config
	config.Init("f", "api_serv", "api_serv.conf")

	err := config.Load(&conf)
	if err != nil {
		return
	}

	logFileName := conf.LogPath + "api_serv.log"
	logFile, err := os.OpenFile(logFileName, os.O_CREATE|os.O_RDWR, 0666)
	if err != nil {
		fmt.Println("Open log file error: ", err)
		return
	}
	log.SetOutput(logFile)
	log.SetOutputLevel(conf.LogLevel)

	hd, err := httpserv.NewService(&conf)
	if err != nil {
		log.Error("erro config ,start fail ", err)
		return
	}

	err = hd.Run()
	if err != nil {
		log.Error("start fail ", err)
	}
}
