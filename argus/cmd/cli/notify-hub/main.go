package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"html"
	"io/ioutil"
	"log"
	"net/http"
)

func main() {

	http.HandleFunc("/bar", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintf(w, "Hello, %q", html.EscapeString(r.URL.Path))
	})

	http.HandleFunc("/json", func(w http.ResponseWriter, r *http.Request) {

		defer r.Body.Close()
		bs, _ := ioutil.ReadAll(r.Body)

		buf := bytes.NewBuffer(nil)
		r.Write(buf)
		fmt.Print(buf.String())

		var m = new(interface{})
		if err := json.Unmarshal(bs, m); err != nil {
			fmt.Printf("BAD JSON BODY: %v %s", err, string(bs))
		} else {
			bs, _ := json.MarshalIndent(m, "\n", "	")
			fmt.Println(string(bs))
		}

		fmt.Println("")

		fmt.Fprintf(w, "Hello, %q", html.EscapeString(r.URL.Path))
	})

	log.Fatal(http.ListenAndServe(":8080", nil))
}
