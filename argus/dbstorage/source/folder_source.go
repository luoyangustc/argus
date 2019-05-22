package source

import (
	"context"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"

	"github.com/pkg/errors"
	xlog "github.com/qiniu/xlog.v1"
	"qiniu.com/argus/dbstorage/proto"
	"qiniu.com/argus/dbstorage/util"
)

var _ ISource = new(FolderSource)

var StopSignalErr = errors.New("receive task stop signal, stop filepath.Walk")

type FolderSource struct {
	folderPath string
}

func NewFolderSource(path string) ISource {
	return &FolderSource{folderPath: path}
}

func (s *FolderSource) Read(ctx context.Context, check func(int) proto.ImageProcess) (<-chan proto.TaskSource, error) {
	xl := xlog.FromContextSafe(ctx)
	ch := make(chan proto.TaskSource)

	go func() {
		i := -1
		err := filepath.Walk(s.folderPath, func(path string, f os.FileInfo, err error) error {
			select {
			case <-ctx.Done():
				//receive stop signal
				return StopSignalErr
			default:
				if f == nil {
					return err
				}

				//skip directory & windows thumb file & hidden file of linux or mac
				if f.IsDir() || strings.ToLower(f.Name()) == "thumb.db" || util.Substring(f.Name(), 0, 1) == "." {
					return nil
				}

				//start to handle normal file
				i++

				process := check(i)
				if process == proto.HANDLED_LAST_TIME {
					//handled by last time, skip
					return nil
				}

				imageContent, err := ioutil.ReadFile(path)
				if err != nil {
					xl.Errorf("error when reading file [%s]: %s", path, err)
					err = proto.ErrOpenImage
				}

				ch <- proto.TaskSource{
					Index:       i,
					Content:     imageContent,
					Id:          proto.ImageId(path),
					URI:         proto.ImageURI(path),
					Process:     process,
					PreCheckErr: err,
				}
			}
			return nil
		})

		if err != nil && err != StopSignalErr {
			xl.Errorf("error when filepath.Walk() on [%s]: %v\n", s.folderPath, err)
		}

		close(ch)
	}()

	return ch, nil
}

func (s *FolderSource) GetInfo(ctx context.Context) (int, error) {
	return 0, nil
}
