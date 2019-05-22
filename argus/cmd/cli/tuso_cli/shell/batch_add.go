package shell

import (
	"bufio"
	"context"
	"io"
	"os"
	"strings"

	"github.com/qiniu/xlog.v1"
	"github.com/spf13/cobra"

	"qiniu.com/argus/tuso/proto"
)

var xl = xlog.NewWith("tuso-cli")

type batchAdd struct {
	cl proto.UserApi
}

func (b *batchAdd) addImages(req *proto.PostImageReq) (resp *proto.PostImageResp, err error) {
	return b.cl.PostImage(context.Background(), req, nil)
}

func (b *batchAdd) parseFile(hub, filePath string) (err error) {
	f, err := os.Open(filePath)
	if err != nil {
		return
	}
	rd := bufio.NewReader(f)
	var line []byte
	successCnt := 0
	errorLine := 0
	notImageLine := 0
	existsLine := 0
	images := make([]entry, 0, 100)

	do := func() error {
		req := new(proto.PostImageReq)
		req.Hub = hub
		req.Op = "ADD"
		for _, img := range images {
			req.Images = append(req.Images, proto.ImageKey{
				Key: img.Key,
			})
		}
		resp, err := b.addImages(req)
		if err != nil {
			if successCnt == 0 {
				return err
			}
			xl.Warn(err)
		} else {
			successCnt += resp.SuccessCnt
			existsLine += resp.ExistsCnt
		}
		images = images[:0]
		return nil
	}

	for {
		line, _, err = rd.ReadLine()
		if err != nil {
			if err != io.EOF {
				xl.Error(err)
			}
			break
		}

		fields := strings.Split(string(line), "\t")
		if len(fields) != 7 {
			if len(line) > 2 {
				errorLine++
			}
			continue
		}
		e := entry{
			Key:      fields[0],
			Fsize:    fields[1],
			Hash:     fields[2],
			PutTime:  fields[3],
			MimeType: fields[4],
			FileType: fields[5],
			EndUser:  fields[6],
		}
		if !strings.HasPrefix(e.MimeType, "image/") {
			notImageLine++
			continue
		}
		images = append(images, e)

		if len(images) == 100 {
			err = do()
			xl.Infof("process ing, success %v, error %v, not image %v, image exists %v", successCnt, errorLine, notImageLine, existsLine)
			if err != nil {
				return err
			}
		}
	}
	do()
	xl.Infof("process over, success %v, error %v, not image %v, image exists %v", successCnt, errorLine, notImageLine, existsLine)
	return nil
}

type entry struct {
	Key      string
	Fsize    string
	Hash     string
	PutTime  string
	MimeType string
	FileType string
	EndUser  string
}

var batchAddCmd = &cobra.Command{
	Use: "add <Hub> <ListBucketResultFile>",
	RunE: func(cmd *cobra.Command, args []string) error {
		if len(args) != 2 {
			return cmd.Help()
		}
		b := new(batchAdd)
		var err error
		b.cl, err = getClient()
		if err != nil {
			return err
		}
		return b.parseFile(args[0], args[1])
	},
}
