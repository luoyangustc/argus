package image_sync

import (
	"testing"
)

func TestAPIDoc(t *testing.T) {

	a := APIDoc{
		Path: "/v1/foo", Version: "V0.1.0",
		Desc: []string{"这是测试"},
		Request: `POST /v1/foo HTTP/1.1
Content-Type: application.json

{
}`,
		Response: `200 ok
Content-Type: appliation/json

{
}`,
		RequestParam: []APIDocParam{
			{Name: "`aa`", Type: "string", Desc: "测试"},
			{Name: "`aa`", Type: "string", Desc: "测试"},
		},
		ResponseParam: []APIDocParam{
			{Name: "`aa`", Type: "string", Desc: "测试"},
			{Name: "`aa`", Type: "string", Desc: "测试"},
		},
		ErrorMessage: []APIDocError{
			{Code: 400401, Desc: "测试"},
			{Code: 400401, Desc: "测试"},
		},
	}
	bs, _ := a.Marshal()

	t.Log(string(bs))
	// t.Fatal(string(bs))

}
