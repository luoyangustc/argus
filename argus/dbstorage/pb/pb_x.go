// +build linux darwin freebsd netbsd openbsd solaris dragonfly
// +build !appengine

package pb

import (
	"os"
	"sync"

	"golang.org/x/sys/unix"
)

var (
	echoLockMutex sync.Mutex
	tty           *os.File
)

func init() {
	echoLockMutex.Lock()
	defer echoLockMutex.Unlock()

	var err error
	tty, err = os.Open("/dev/tty")
	if err != nil {
		tty = os.Stdin
	}
}

// terminalWidth returns width of the terminal.
func terminalWidth() (int, error) {
	echoLockMutex.Lock()
	defer echoLockMutex.Unlock()

	fd := int(tty.Fd())

	ws, err := unix.IoctlGetWinsize(fd, unix.TIOCGWINSZ)
	if err != nil {
		return 0, err
	}

	return int(ws.Col), nil
}
