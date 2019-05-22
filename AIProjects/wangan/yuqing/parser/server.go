package parser

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"html/template"
	"net/http"
	"strconv"
	"strings"
	"time"

	xlog "github.com/qiniu/xlog.v1"
	"qiniu.com/argus/AIProjects/wangan/yuqing"
	"qiniu.com/argus/com/uri"

	"qbox.us/net/httputil"
)

type M map[string]interface{}

type ResultReq struct {
	CmdArgs []string
	Date    string `json:"date"`
}

type resultHtml struct {
	Results []struct {
		Op      string
		Class   string
		Score   float32
		Video   string
		Message string
	}
}

func (p *parser) GetResult(w http.ResponseWriter, req *http.Request) {
	req.ParseForm()
	date := req.FormValue("date")
	ty := req.FormValue("type")

	start, err := time.Parse("20060102", date)
	if err != nil {
		httputil.ReplyError(w, "invalid date:"+date, http.StatusBadRequest)
		return
	}
	end := start.Add(24 * time.Hour)

	coll := p.coll.CopySession()
	defer coll.CloseSession()
	var results []yuqing.Result
	query := M{"ops": M{"$exists": true}, "parseTime": M{"$gt": start, "$lt": end}, "error": M{"$exists": false}}
	if ty != "" {
		tt, _ := strconv.Atoi(ty)
		query["type"] = tt
	}
	if err = coll.Find(&query).All(&results); err != nil {
		httputil.ReplyError(w, "query db:"+err.Error(), http.StatusInternalServerError)
		return
	}

	var html resultHtml
	illegel := []string{"fight_police", "fight_person", "march_banner", "march_crowed"}
	for _, result := range results {

		for name, op := range result.Ops {
			var (
				classes []string
			)

			for _, label := range op.Labels {
				for _, ille := range illegel {
					if label.Name == ille {
						classes = append(classes, label.Name)
					}
				}
			}
			if len(classes) == 0 {
				continue
			}
			buf := bytes.NewBuffer([]byte{})
			encoder := json.NewEncoder(buf)
			encoder.SetEscapeHTML(false)
			encoder.Encode(result.Message)
			html.Results = append(html.Results, struct {
				Op      string
				Class   string
				Score   float32
				Video   string
				Message string
			}{
				Op:      name,
				Class:   strings.Join(classes, ","),
				Score:   result.Score,
				Video:   result.URI,
				Message: buf.String(),
			})
		}
	}

	t, _ := template.ParseFiles("result.html")
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	_ = t.Execute(w, html)

	return
}

// /proxy?uri=<base64_qiniu_uri>
func (p *parser) Proxy(w http.ResponseWriter, req *http.Request) {
	var (
		ctx = context.Background()
		xl  = xlog.FromContextSafe(ctx)
	)
	req.ParseForm()
	u := req.FormValue("uri")

	bs, err := base64.URLEncoding.DecodeString(u)
	if err != nil || len(bs) == 0 {
		httputil.ReplyError(w, "bad uri", http.StatusBadRequest)
		return
	}

	resp, err := p.handler.Get(ctx, uri.Request{
		URI: string(bs),
	})

	if err != nil || resp == nil || resp.Size == 0 {
		xl.Errorf("fail to fetch uri (%s), error: %v, resp: %v", u, err, resp)
		httputil.ReplyError(w, "fail to fetch uri", http.StatusInternalServerError)
		return
	}
	defer resp.Body.Close()
	for k, v := range resp.Header {
		w.Header().Set(k, strings.Join(v, ";"))
	}
	httputil.ReplyBinary(w, resp.Body, resp.Size)
}
