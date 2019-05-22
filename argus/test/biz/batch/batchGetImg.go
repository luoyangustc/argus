package batch

import (
	"encoding/json"
	"io/ioutil"
	"os"
	"strconv"

	// . "github.com/onsi/ginkgo"
	// . "github.com/onsi/gomega"

	E "qiniu.com/argus/test/biz/env"
	"qiniu.com/argus/test/configs"
)

var BatchSize = 50

type picInfo struct {
	Name     string `json:"key"`
	Hash     string `json:"hash"`
	Fsize    int32  `json:"fsize"`
	MimeType string `json:"mimeType"`
	Puttime  int64  `json:"putTime"`
	Type     int    `json:"type"`
	Status   int    `json:"status"`
}

type picList struct {
	Marker string    `json:"marker"`
	Imgs   []picInfo `json:"items"`
}

type Imgfile struct {
	Imgname string
	Imgurl  string
}

// var _ = Describe("terror", func() {
// 	var path = "/v1/terror"
// 	var client = E.Env.GetClientArgus()
// 	var params = map[string]bool{"detail": true}
// 	Context("get", func() {
// 		It("new", func() {
// 			imgList, err := BatchGetImg("normal")
// 			Expect(err).To(BeNil())
// 			for _, img := range imgList {
// 				fmt.Println(img.Imgname)
// 				Expect(biz.StoreUri(img.Imgname, img.Imgurl)).Should(BeNil())
// 				resp, err := client.PostWithJson(path,
// 					proto.NewArgusReq(img.Imgurl, nil, params))
// 				Expect(err).Should(BeNil())
// 				Expect(resp.Status()).To(Or(Equal(200), BeNumerically("<", 500)))
// 				var resObj proto.ArgusRes
// 				err = json.Unmarshal(resp.BodyToByte(), &resObj)
// 				Expect(err).Should(BeNil())
// 			}
// 		})
// 	})
// })

func BatchGetImg(detectTpye string) ([]Imgfile, error) {
	var image []Imgfile
	if os.Getenv("TEST_ENV") == "" {
		imgSet := "test/image/" + detectTpye + "/"
		dirList, err := ioutil.ReadDir("testdata/" + imgSet)
		if err != nil {
			return nil, err
		}
		for _, v := range dirList {
			var img Imgfile
			img.Imgname = v.Name()
			img.Imgurl = E.Env.GetURIPrivate(imgSet + v.Name())
			image = append(image, img)
		}
	} else {
		var resObj picList
		var clientBucket = E.Env.GetClientBucket()
		prefix := "test/image/"
		var p = "/list?bucket=" + configs.Configs.Atservingprivatebucketz0.Name + "&limit=" + strconv.Itoa(BatchSize) + "&prefix=" + prefix + detectTpye
		resp, err := clientBucket.GetT(p, "application/x-www-form-urlencoded")
		if err != nil {
			return nil, err
		}
		err2 := json.Unmarshal(resp.BodyToByte(), &resObj)
		if err2 != nil {
			return nil, err2
		}
		for _, v := range resObj.Imgs {
			var img Imgfile
			img.Imgname = v.Name
			img.Imgurl = E.Env.GetURIPrivate(v.Name)
			image = append(image, img)
		}
	}
	return image, nil
}

func GetBucket(prefix string) ([]Imgfile, error) {
	var image []Imgfile
	if os.Getenv("TEST_ENV") == "" {
		imgSet := prefix
		dirList, err := ioutil.ReadDir("testdata/" + imgSet)
		if err != nil {
			return nil, err
		}
		for _, v := range dirList {
			var img Imgfile
			img.Imgname = v.Name()
			img.Imgurl = E.Env.GetURIPrivate(imgSet + v.Name())
			image = append(image, img)
		}
	} else {
		var resObj picList
		var clientBucket = E.Env.GetClientBucket()
		var p = "/list?bucket=" + configs.Configs.Atservingprivatebucketz0.Name + "&limit=" + strconv.Itoa(BatchSize) + "&prefix=" + prefix
		resp, err := clientBucket.GetT(p, "application/x-www-form-urlencoded")
		if err != nil {
			return nil, err
		}
		err2 := json.Unmarshal(resp.BodyToByte(), &resObj)
		if err2 != nil {
			return nil, err2
		}
		for _, v := range resObj.Imgs {
			var img Imgfile
			img.Imgname = v.Name
			img.Imgurl = E.Env.GetURIPrivate(v.Name)
			image = append(image, img)
		}
	}
	return image, nil
}

func BatchGetVideo(prefix string) ([]Imgfile, error) {
	var image []Imgfile
	if os.Getenv("TEST_ENV") == "" {
		var imgSet = prefix
		dirList, err := ioutil.ReadDir("testdata/argusvideo/" + imgSet)
		if err != nil {
			return nil, err
		}
		for _, v := range dirList {
			var img Imgfile
			img.Imgname = v.Name()
			img.Imgurl = E.Env.GetURIVideo(imgSet + v.Name())
			image = append(image, img)
		}
	} else {
		var resObj picList
		var clientBucket = E.Env.GetClientBucket()
		var p = "/list?bucket=" + configs.Configs.Atservingprivatebucketz0.Name + "&limit=" + strconv.Itoa(BatchSize) + "&prefix=" + prefix
		resp, err := clientBucket.GetT(p, "application/x-www-form-urlencoded")
		if err != nil {
			return nil, err
		}
		err2 := json.Unmarshal(resp.BodyToByte(), &resObj)
		if err2 != nil {
			return nil, err2
		}
		for _, v := range resObj.Imgs {
			var img Imgfile
			img.Imgname = v.Name
			img.Imgurl = E.Env.GetURIPrivate(v.Name)
			image = append(image, img)
		}
	}
	return image, nil
}
