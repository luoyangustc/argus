package teapot

import (
	"fmt"
	"html"
	"net/http"
	"os"
	"strings"

	"github.com/teapots/inject"
)

const (
	panicHtml = `<html>
<head><title>PANIC: %s</title>
<style type="text/css">
html, body {
	font-family: "Roboto", sans-serif;
	color: #333333;
	background-color: #3e3d49;
	margin: 0px;
}
h1 {
	color: #ffffff;
	background-color: #d04526;
	padding: 20px;
}
pre.message {
	font-weight: bold;
}
pre.message,
.stack {
	font-family: Menlo, monospace;
	font-size: 13px;
	margin: 25px 20px;
	padding: 15px;
	background-color: #ffffff;
}
.stack .line {
	margin: 2px 0;
}
.stack .odd {
	margin: 15px 0 5px;
}
.stack .even {
	margin: 5px 0 15px 2em;
}
.stack .file {
	font-weight: bold;
}
</style>
</head><body>
<h1>PANIC: %s</h1>
<pre class="message">%T: %#v</pre>
<div class="stack">%s</div>
</body>
</html>`
)

// Recovery returns a middleware that recovers from any panics and writes a 500 if there was one.
// While app in development mode, Recovery will also output the panic as HTML.
func RecoveryFilter() inject.Provider {
	return func(ctx Context, config *Config) {
		defer func() {
			if err := recover(); err != nil {
				stack := string(Stack(5))

				var log Logger
				fErr := ctx.Find(&log, "")
				if fErr != nil {
					fmt.Fprintf(os.Stderr, "PANIC: %#v FindLog:%v\n%s", err, fErr, stack)
					return
				}

				log.Errorf("PANIC: %#v\n%s", err, stack)

				var rw http.ResponseWriter
				ctx.Find(&rw, "")

				trw := rw.(ResponseWriter)
				if trw.Written() {
					return
				}

				// in prod mode only set status code
				// fall back to other filter
				rw.WriteHeader(http.StatusInternalServerError)

				// respond with panic message while in development mode
				var body []byte
				if config.RunMode.IsDev() {
					rw.Header().Set("Content-Type", "text/html; charset=utf-8")
					body = []byte(fmt.Sprintf(panicHtml, err, err, err, err, styleStack(stack)))
					rw.Write(body)
				}
			}
		}()

		ctx.Next()
	}
}

func styleStack(stack string) string {
	stack = html.EscapeString(stack)
	lines := strings.Split(stack, "\n")
	for i, line := range lines {
		if len(line) == 0 {
			continue
		}

		if line[0] == '/' {
			parts := strings.Split(line, " (")
			if len(parts) >= 2 {
				parts[0] = fmt.Sprintf(`<span class="file">%s</span> (`, parts[0])
				line = strings.Join(parts, "")
			}
			line = fmt.Sprintf(`<div class="line odd">%s</div>`, line)
		} else {
			line = fmt.Sprintf(`<div class="line even">%s</div>`, line)
		}

		lines[i] = line
	}
	stack = strings.Join(lines, "\n")
	return stack
}
