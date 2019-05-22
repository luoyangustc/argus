package serving

import (
	"encoding/json"
	"fmt"
	"testing"
)

func TestParse(t *testing.T) {
	var request struct {
		Feautres []float32Feature `json:"features"`
	}
	//var fs _FeatureSearch
	ff := float32Feature{Value: []float32{1.0, 2.0}}
	request.Feautres = append(request.Feautres, ff)
	fmt.Printf("%#v\n", &request)
	_, err := json.Marshal(&request)
	fmt.Println(err)
}
