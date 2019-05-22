package bucket

import (
	"testing"
)

func TestBucket(t *testing.T) {

	// 示例

	// b := Bucket{Config: Config{
	// 	Config: kodo.Config{
	// 		AccessKey: "",
	// 		SecretKey: "",
	// 		RSHost:    "http://10.200.20.25:12501",
	// 		IoHost:    "10.200.20.23",
	// 	},
	// 	Bucket: "argus-bcp",
	// 	Domain: "",
	// }}

	// cli := ahttp.NewQiniuAuthRPCClient(b.Config.Config.AccessKey, b.Config.Config.SecretKey, time.Second*10)
	// var domains = []struct {
	// 	Domain string `json:"domain"`
	// 	Tbl    string `json:"tbl"`
	// 	Global bool   `json:"global"`
	// }{}
	// err := cli.Call(context.Background(), &domains,
	// 	"GET", fmt.Sprintf("http://10.200.20.23:12500/v7/domain/list?tbl=%s", b.Config.Bucket),
	// )
	// log.Infof("%#v", domains)

	// b.Config.Domain = domains[0].Domain

	// rs, err := b.ReadByDomain(context.Background(), "20180528104137_20180528104138__")
	// if err != nil {
	// 	t.Fatalf("read failed. %v", err)
	// }
	// defer rs.Close()

	// bs, _ := ioutil.ReadAll(rs)
	// t.Errorf("LEN: %d", len(bs))

}
