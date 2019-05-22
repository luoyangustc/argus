package serving

import (
	"time"
)

type EvalConfig struct {
	Host string `json:"host"`
	URL  string `json:"url"`
	// timeout: second
	Timeout time.Duration `json:"timeout"`
}
