package largefile

import (
	"os"
	"time"
)

// --------------------------------------------------------------------

var zeroTime time.Time

type fileInfo struct {
	fsize int64
}

func (p *fileInfo) Name() string {
	return ""
}

func (p *fileInfo) Size() int64 {
	return p.fsize
}

func (p *fileInfo) Mode() os.FileMode {
	return 0666
}

func (p *fileInfo) ModTime() time.Time {
	return zeroTime
}

func (p *fileInfo) IsDir() bool {
	return false
}

func (p *fileInfo) Sys() interface{} {
	return nil
}

// --------------------------------------------------------------------
