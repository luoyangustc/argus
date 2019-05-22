package source

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"

	"github.com/pkg/errors"
	xlog "github.com/qiniu/xlog.v1"
	"qiniu.com/argus/dbstorage/proto"
)

var _ ISource = new(JsonSource)

type JsonSource struct {
	fileContent []byte
}

type jsonLine struct {
	Image struct {
		Id   string `json:"id"`
		Uri  string `json:"uri"`
		Tag  string `json:"tag"`
		Desc string `json:"desc"`
	} `json:"image"`
}

func NewJsonSource(content []byte) ISource {
	return &JsonSource{fileContent: content}
}

func (s *JsonSource) Read(ctx context.Context, check func(int) proto.ImageProcess) (<-chan proto.TaskSource, error) {
	ch := make(chan proto.TaskSource)

	go func() {
		realLineNum := 0
		nonEmptyLineIndex := -1
		scanner := bufio.NewScanner(bytes.NewReader(s.fileContent))
		for scanner.Scan() {
			realLineNum++
			text := scanner.Text()
			if text == "" {
				continue
			}
			select {
			case <-ctx.Done():
				//receive stop signal
				close(ch)
				return
			default:
				nonEmptyLineIndex++

				process := check(nonEmptyLineIndex)
				if process == proto.HANDLED_LAST_TIME {
					//handled by last time, skip
					continue
				}

				var line jsonLine
				_ = json.Unmarshal([]byte(text), &line)
				task := proto.TaskSource{
					Index:   nonEmptyLineIndex,
					Id:      proto.ImageId(line.Image.Id),
					URI:     proto.ImageURI(line.Image.Uri),
					Tag:     proto.ImageTag(line.Image.Tag),
					Desc:    proto.ImageDesc(line.Image.Desc),
					Process: process,
				}
				ch <- task
			}
		}
		close(ch)
	}()

	return ch, nil
}

func (s *JsonSource) GetInfo(ctx context.Context) (int, error) {
	xl := xlog.FromContextSafe(ctx)
	realLineNum := 0
	nonEmptyLineNum := 0
	scanner := bufio.NewScanner(bytes.NewReader(s.fileContent))
	for scanner.Scan() {
		realLineNum++
		text := scanner.Text()
		if text == "" {
			continue
		}
		nonEmptyLineNum++
		var line jsonLine
		err := json.Unmarshal([]byte(text), &line)
		if err != nil {
			err := errors.Errorf("line %d : not correct json format", realLineNum)
			xl.Error(err)
			return 0, err
		}
		if line.Image.Id == "" || line.Image.Uri == "" {
			err := errors.Errorf("line %d : must have node 'image', and have nodes 'id' and 'uri' inside", realLineNum)
			xl.Error(err)
			return 0, err
		}
	}

	return nonEmptyLineNum, nil
}
