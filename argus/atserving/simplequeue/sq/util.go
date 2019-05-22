package sq

import (
	"log"
	"os"

	"github.com/nsqio/go-nsq"
)

type logger interface {
	Output(calldepth int, s string) error
}

var Logger logger = log.New(os.Stderr, "", log.Flags())
var LogLevel nsq.LogLevel = nsq.LogLevelInfo
