package evaltest

import (
	"bufio"
	"bytes"
	"encoding/base64"
	"encoding/json"
	"strings"

	K "github.com/onsi/ginkgo"
	O "github.com/onsi/gomega"
	C "qiniu.com/argus/test/biz/client"
	"qiniu.com/argus/test/biz/env"
	"qiniu.com/argus/test/biz/proto"
	T "qiniu.com/argus/test/biz/tsv"
)

type AdsClassifierRes struct {
	Code    int
	Message string
	Result  map[string]struct {
		Confidences []AdsClassifierConfidence
		Summary     AdsClassifierSummary
	}
}

type AdsClassifierSummary struct {
	Label string
	Score float64
}

type AdsClassifierConfidence struct {
	Keys  []string
	Label string
	Score float64
}

type Data struct {
	Text []string `json:"text"`
}

func TextToBase64(Texts []string) (string, error) {
	buf, err := json.Marshal(Texts)
	if err != nil {
		return "", err
	}
	uriBase64 := "data:application/octet-stream;base64," + base64.StdEncoding.EncodeToString(buf)
	return uriBase64, nil
}

func GetSummary(confidences []AdsClassifierConfidence) (summary AdsClassifierSummary) {
	for _, conf := range confidences {
		if len(conf.Keys) != 0 && conf.Score > summary.Score {
			summary.Score = conf.Score
			summary.Label = conf.Label
		}
	}
	if summary.Label == "" {
		summary.Label = "normal"
		summary.Score = 1.0
	}
	return
}

func TsvClassifierTest(client C.Client, t T.Tsv, Check func([]byte, []byte, float64)) {
	buf, err := env.Env.GetTSV(t.Name)
	if err != nil {
		K.By("tsv下载失败：")
		panic(err)
	}
	// defer read.Close()
	reader := bufio.NewReader(bytes.NewReader(buf))
	// reader := bufio.NewReader(read)
	scanner := bufio.NewScanner(reader)
	scanner.Split(bufio.ScanLines)
	for scanner.Scan() {
		line := scanner.Text()
		record := strings.Split(line, "\t")
		var data Data
		textdata := record[0]
		if err1 := json.Unmarshal([]byte(textdata), &data); err1 != nil {
			panic(err1)
		}
		uri, err := TextToBase64(data.Text)
		if err != nil {
			panic(err)
		}
		K.It("测试:"+record[0], func() {
			K.By("tsv文件结果:\n" + line)
			resp, err := client.PostWithJson(t.Path,
				proto.NewArgusReq(uri, nil, t.Params))
			O.Expect(err).Should(O.BeNil())
			if resp.Status() == 200 {
				if Check == nil {
					return
				}
				Check(resp.BodyToByte(), []byte(record[1]), t.Precision)
			} else {
				T.CheckError(resp.BodyToByte(), []byte(record[1]))
			}
		})
	}
}
