package main

import (
	"flag"
	"runtime"

	"qbox.us/cc/config"
)

func main() {

	runtime.GOMAXPROCS(runtime.NumCPU())

	config.Init("f", "bucket-proxy", "bucket-proxy.conf")
	daemon := flag.Bool("d", false, "as daemon")
	flag.Parse()

	if *daemon {
		serverMain()
		return
	}

	syncMain(flag.Args()[:])

}
