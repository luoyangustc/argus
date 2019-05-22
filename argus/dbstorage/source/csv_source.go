package source

import (
	"bytes"
	"context"
	"encoding/csv"
	"strings"

	"github.com/pkg/errors"
	xlog "github.com/qiniu/xlog.v1"
	"qiniu.com/argus/dbstorage/proto"
)

var _ ISource = new(CsvSource)

type CsvSource struct {
	fileContent []byte
}

func NewCsvSource(content []byte) ISource {
	return &CsvSource{fileContent: content}
}

func (s *CsvSource) Read(ctx context.Context, check func(int) proto.ImageProcess) (<-chan proto.TaskSource, error) {
	xl := xlog.FromContextSafe(ctx)
	ch := make(chan proto.TaskSource)

	lines, err := csv.NewReader(bytes.NewReader(s.fileContent)).ReadAll()
	if err != nil {
		xl.Errorf("read task csv file failed: %s", err)
		return nil, errors.Errorf("invalid csv file, %s", err.Error())
	}

	//verify the column of csv file
	if len(lines) > 0 && len(lines[0]) < 2 {
		err = errors.New("csv file must contains at least two columns for id and uri")
		xl.Error(err)
		return nil, err
	}

	go func() {
		i := -1
		for {
			select {
			case <-ctx.Done():
				//receive stop signal
				close(ch)
				return
			default:
				i++
				if i >= len(lines) {
					//finish reading all file
					close(ch)
					return
				}

				process := check(i)
				if process == proto.HANDLED_LAST_TIME {
					//handled by last time, skip
					continue
				}

				data := lines[i]
				var id, uri, tag, desc string
				id = strings.TrimSpace(data[0])
				uri = strings.TrimSpace(data[1])
				if len(data) > 2 {
					tag = strings.TrimSpace(data[2])
				}
				if len(data) > 3 {
					desc = strings.TrimSpace(data[3])
				}
				ch <- proto.TaskSource{
					Index:   i,
					Id:      proto.ImageId(id),
					URI:     proto.ImageURI(uri),
					Tag:     proto.ImageTag(tag),
					Desc:    proto.ImageDesc(desc),
					Process: process,
				}
			}
		}
	}()

	return ch, nil
}

func (s *CsvSource) GetInfo(ctx context.Context) (int, error) {
	xl := xlog.FromContextSafe(ctx)
	lines, err := csv.NewReader(bytes.NewReader(s.fileContent)).ReadAll()
	if err != nil {
		xl.Errorf("read task csv file failed: %s", err)
		return 0, errors.Errorf("invalid csv file, %s", err.Error())
	}

	//verify the column of csv file
	if len(lines) > 0 && len(lines[0]) < 2 {
		err = errors.New("csv file of the task must contains at least two columns for id and uri")
		xl.Error(err)
		return 0, err
	}

	return len(lines), nil
}
