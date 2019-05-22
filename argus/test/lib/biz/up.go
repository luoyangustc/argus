package biz

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strconv"

	c "qiniu.com/argus/test/configs"
	"qiniu.com/argus/test/lib/auth"
	"qiniu.com/argus/test/lib/qnhttp"
)

func FormUp(key, filepath, token string) (*qnhttp.Response, error) {
	body := map[string]string{"key": key, "filepath": filepath, "token": token}

	s := qnhttp.New().Set("Content-Type", "multipart/form-data")
	resp, err := s.Post(c.Configs.Host.UP, body, nil, nil)
	return resp, err
}

// 用于普通上传
func DoFormUp(user auth.AccessInfo, bucket, upKey, filePath string) (resp *qnhttp.Response, err error) {
	//签token
	putPolicy := &auth.PutPolicy{
		Scope: bucket,
	}
	upToken := putPolicy.MakeUptoken(user.Key, user.Secret)
	resp, err = FormUp(upKey, filePath, upToken)
	if err != nil {
		println(err.Error())
		return nil, err
	}
	return
}

func CallbackFormUp(key, filepath, token, callbackurl string) (*qnhttp.Response, error) {
	type callBackUrlStruct struct {
		Url string `json:"url"`
	}
	u := callBackUrlStruct{Url: callbackurl}
	cul := struct {
		Callback callBackUrlStruct `json:"callback"`
	}{Callback: u}
	out, _ := json.Marshal(&cul)
	body := map[string]string{"key": key, "filepath": filepath, "token": token, "x:vod": string(out)}

	s := qnhttp.New().Set("Content-Type", "multipart/form-data")
	resp, err := s.Post(c.Configs.Host.UP, body, nil, nil)
	return resp, err
}

type MkblkArgs struct {
	User    auth.AccessInfo
	Data    []byte
	Bucket  string
	BlkSize int64
	Token   string
}

func Mkblk(args MkblkArgs) (*qnhttp.Response, error) {
	dataSize := len(args.Data)

	blkSize := args.BlkSize
	if blkSize == 0 {
		blkSize = int64(dataSize)
	}

	url := c.Configs.Host.UP + "/mkblk/" + strconv.FormatInt(blkSize, 10)
	s := qnhttp.New()
	s.Header.Set("Content-Type", "application/octet-stream")
	s.Header.Set("Content-Length", strconv.Itoa(dataSize))

	var uptoken string
	if args.Token != "" {
		uptoken = args.Token
	} else {
		uptoken = auth.SignUptoken(args.User, args.Bucket)
	}

	s.Header.Add("Authorization", "UpToken "+uptoken)

	resp, err := s.Post(url, string(args.Data), nil, nil)
	return resp, err
}

type BputArgs struct {
	Ctx             string
	NextChunkOffset int64
	Host            string
	Token           string
	Body            []byte
}

func Bput(args BputArgs) (*qnhttp.Response, error) {
	url := args.Host + "/bput/" + args.Ctx + "/" + fmt.Sprintf("%v", args.NextChunkOffset)
	s := qnhttp.New()
	s.Header.Set("Content-Type", "application/octet-stream")
	s.Header.Add("Authorization", "UpToken "+args.Token)

	resp, err := s.Post(url, string(args.Body), nil, nil)
	return resp, err
}

type BlkputRet struct {
	Ctx      string `json:"ctx"`
	Checksum string `json:"checksum"`
	Crc32    uint32 `json:"crc32"`
	Offset   int64  `json:"offset"`
	Host     string `json:"host"`
}

func ResumeUploadBlock(f *os.File, token string, offset int64, fsize int64) (string, string) {
	chunkSize := 1024 * 1024 // 1M

	mkblkArgs := MkblkArgs{
		Token:   token,
		BlkSize: fsize,
	}

	var b []byte
	if fsize <= int64(chunkSize) {
		b = make([]byte, fsize)
	} else {
		b = make([]byte, chunkSize)
	}

	_, err := f.ReadAt(b, offset)
	if err != nil && err != io.EOF {
		fmt.Println("ERROR: ReadAt failed!")
		panic(err)
	}

	mkblkArgs.Data = b

	res, _ := Mkblk(mkblkArgs)
	if res.Status() != 200 {
		fmt.Println("ERROR: Make block failed!")
		panic(res)
	}

	mkblkRes := BlkputRet{}
	if err := res.Unmarshal(&mkblkRes); err != nil {
		panic(err)
	}

	if fsize <= int64(chunkSize) {
		return mkblkRes.Ctx, mkblkRes.Host
	}

	// bput upload
	bputArgs := BputArgs{
		Ctx:             mkblkRes.Ctx,
		Host:            mkblkRes.Host,
		Token:           token,
		NextChunkOffset: mkblkRes.Offset,
	}

	i := 1
	for {
		rest := mkblkArgs.BlkSize - int64(chunkSize*i)
		if rest < 0 {
			break
		}

		var temp []byte
		if rest > int64(chunkSize) {
			temp = make([]byte, chunkSize)

		} else {
			temp = make([]byte, rest)
		}

		_, err := f.ReadAt(temp, bputArgs.NextChunkOffset)
		if err != nil && err != io.EOF {
			fmt.Println("ERROR: ReadAt failed!")
		}
		bputArgs.Body = temp
		res, _ = Bput(bputArgs)
		if res.Status() != 200 {
			fmt.Println("ERROR: Bput failed!")
		}

		tempBputRes := new(BlkputRet)
		if err := res.Unmarshal(&tempBputRes); err != nil {
			panic(err)
		}
		bputArgs.NextChunkOffset = tempBputRes.Offset
		bputArgs.Ctx = tempBputRes.Ctx
		i++
	}

	return bputArgs.Ctx, mkblkRes.Host
}
