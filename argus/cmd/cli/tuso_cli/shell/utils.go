package shell

import (
	"bytes"
	"encoding/json"
	"fmt"
	"html/template"
	"log"
	"strings"
	"text/tabwriter"
)

func renderIdent(tmpl string, data interface{}) string {
	funcMap := template.FuncMap{
		"json": func(v interface{}) string {
			buf, _ := json.Marshal(v)
			s := string(buf)
			if s == "null" {
				return ""
			}
			return s
		},
	}

	tmpl = strings.TrimSpace(tmpl)
	t, err := template.New("").Funcs(funcMap).Parse(tmpl)
	if err != nil {
		log.Panicln("bad template", err)
	}
	buf := bytes.Buffer{}
	w := tabwriter.NewWriter(&buf, 0, 8, 6, ' ', 0)
	t.Execute(w, data)
	w.Flush()
	out := buf.String()
	if !strings.HasSuffix(out, "\n") {
		out = fmt.Sprintln(out)
	}
	return out
}
