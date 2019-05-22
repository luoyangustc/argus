package test

import (
	"bufio"
	"bytes"
	"fmt"
	"strconv"
	"strings"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	"qiniu.com/argus/test/biz/assert"
	P "qiniu.com/argus/test/biz/batch"
	E "qiniu.com/argus/test/biz/env"
	"qiniu.com/argus/test/biz/proto"
	biz "qiniu.com/argus/test/biz/util"
	"qiniu.com/argus/test/configs"
)

var _ = Describe("[argus][image]/v3/censor/image", func() {
	var (
		censorClient     = E.Env.GetClientArgus()
		server           = "censorimage"
		censorAPIImgPath = biz.GetPath(server, "image", "path")
		typePulp         = configs.StubConfigs.Servers.Online.ImagePulp
		typeTerror       = configs.StubConfigs.Servers.Online.ImageTerror
		typePolitician   = configs.StubConfigs.Servers.Online.ImagePolitician
		typeAds          = configs.StubConfigs.Servers.Online.ImageAds
		tsvName          = configs.StubConfigs.Servers.Type["image"][server].Tsv
		censorType       []string
	)
	if typePolitician {
		fmt.Println("chose politician")
		censorType = append(censorType, "politician")
	}
	if typeTerror {
		fmt.Println("chose terror")
		censorType = append(censorType, "terror")
	}
	if typePulp {
		fmt.Println("chose pulp")
		censorType = append(censorType, "pulp")
	}
	if typeAds {
		fmt.Println("chose ads")
		censorType = append(censorType, "ads")
	}
	Params := map[string][]string{"scenes": censorType}
	Describe("图片识别", func() {
		Context("/v3/image/censor|三剑", func() {
			buf, err := E.Env.GetTSV(tsvName)
			if err != nil {
				By("tsv下载失败：\n")
				panic(err)
			}
			reader := bufio.NewReader(bytes.NewReader(buf))
			scanner := bufio.NewScanner(reader)
			scanner.Split(bufio.ScanLines)
			var fileList []string
			for scanner.Scan() {
				line := scanner.Text()
				record := strings.Split(line, "\t")
				fileName := configs.StubConfigs.Servers.Type["image"][server].Set + record[0] + "/" + record[1]
				uri := E.Env.GetURIPrivate(fileName)
				if err := biz.StoreUri(fileName, uri); err != nil {
					panic(err)
				}
				fmt.Println(fileName)
				fileList = append(fileList, uri)
			}
			var i = 0
			reader = bufio.NewReader(bytes.NewReader(buf))
			scanner = bufio.NewScanner(reader)
			scanner.Split(bufio.ScanLines)
			for scanner.Scan() {
				uri := fileList[i]
				i++
				line := scanner.Text()
				record := strings.Split(line, "\t")
				desc := "测试case" + strconv.Itoa(i) + ":" + record[0] + "-" + record[2]
				By("tsv文件当前行数据：\n" + line)
				switch record[0] {
				case "pulp":
					if !typePulp {
						break
					}
					It("censor_pulp:"+desc, func() {
						resp, err := GetCensorResp(censorClient, censorAPIImgPath, []string{record[0]}, uri)
						Expect(err).Should(BeNil())
						CheckPulp(resp, record[2], record[3])
					})

				case "politician":
					if !typePolitician {
						break
					}
					It("censor_politician:"+desc, func() {
						resp, err := GetCensorResp(censorClient, censorAPIImgPath, []string{record[0]}, uri)
						Expect(err).Should(BeNil())
						scenes := CheckPolitician(resp, record[2], record[3], record[4])
						if record[2] != "normal" {
							Expect(len(scenes.Details)).Should(BeNumerically(">=", 1))
						}
					})
				case "terror":
					if !typeTerror {
						break
					}
					It("censor_terror:"+desc, func() {
						resp, err := GetCensorResp(censorClient, censorAPIImgPath, []string{record[0]}, uri)
						Expect(err).Should(BeNil())
						CheckTerror(resp, record[2], record[4], record[3])
					})
				case "ads":
					if !typeAds {
						break
					}
					It("censor_ads:"+desc, func() {
						resp, err := GetCensorResp(censorClient, censorAPIImgPath, []string{record[0]}, uri)
						Expect(err).Should(BeNil())
						CheckAds(resp, record[2], record[3])
					})
				default:
					break
				}
			}
		})
	})
	Describe("反向用例", func() {
		Context("错误码校验", func() {
			assert.CheckImageCode(censorClient, censorAPIImgPath, Params)
		})
	})
	Describe("图片库测试", func() {
		Context("normal", func() {
			imgList, err := P.BatchGetImg("normal")
			if err != nil {
				panic(err)
			}
			for _, img := range imgList {
				imgName := img.Imgname
				imgUrl := img.Imgurl
				if imgName == "difftype-4ch.webp" {
					continue
				}
				err = biz.StoreUri(imgName, imgUrl)
				if err != nil {
					panic(err)
				}
				It("测试图片:"+imgName, func() {
					resp, err := censorClient.PostWithJson(censorAPIImgPath,
						proto.NewArgusReq(imgUrl, nil, Params))
					Expect(err).Should(BeNil())
					Expect(resp.Status()).To(Or(Equal(200), BeNumerically("<", 500)))
				})
			}
		})
		Context("特殊图片格式", func() {
			fileList := []string{"pulpsexy.webp", "test-pulp7.bmp"}
			for _, file := range fileList {
				fileName := file
				By("测试图片: " + fileName)
				uri := E.Env.GetURIPrivate(fileName)
				if err := biz.StoreUri(fileName, uri); err != nil {
					panic(err)
				}
				It("特殊图片 "+fileName, func() {
					resp, err := censorClient.PostWithJson(censorAPIImgPath, proto.NewArgusReq(uri, nil, Params))
					Expect(err).Should(BeNil())
					Expect(resp.Status()).To(Equal(200))
				})
			}
		})
	})
})
