package records

import (
	"context"
	"io"
	"os"
	"path"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"

	"github.com/qiniu/xlog.v1"

	"qiniu.com/argus/argus/com/bucket"
)

func TestFile(t *testing.T) {

	{
		dir := path.Join(os.TempDir(), xlog.GenReqId())
		os.MkdirAll(dir, 0755)
		defer os.RemoveAll(dir)
		file := NewFile(bucket.NewLocalFS(dir), 0, 0)
		file.Flush(context.Background())

		scan := file.Scan(context.Background())
		_, _, err := scan.Scan(context.Background())
		assert.Error(t, io.EOF, err)
	}
	{
		dir := path.Join(os.TempDir(), xlog.GenReqId())
		os.MkdirAll(dir, 0755)
		defer os.RemoveAll(dir)
		file := NewFile(bucket.NewLocalFS(dir), time.Hour, 16)
		file.Append(context.Background(), "AAAAAAAA", []byte("XXXXXXXXXXXXXXXX"))
		file.Append(context.Background(), "AAAAAAAB", []byte("XXXXXXXXXXXXXXXX"))
		file.Append(context.Background(), "AAAAAAAC", []byte("XXXXXXXXXXXXXXXX"))
		file.Append(context.Background(), "AAAAAAAD", []byte("XXXXXXXXXXXXXXXX"))
		file.Flush(context.Background())

		scan := file.Scan(context.Background())
		key, _, err := scan.Scan(context.Background())
		assert.NoError(t, err)
		assert.Equal(t, RecordKey("AAAAAAAA"), key)
		key, _, err = scan.Scan(context.Background())
		assert.NoError(t, err)
		assert.Equal(t, RecordKey("AAAAAAAB"), key)
		key, _, err = scan.Scan(context.Background())
		assert.NoError(t, err)
		assert.Equal(t, RecordKey("AAAAAAAC"), key)
		key, _, err = scan.Scan(context.Background())
		assert.NoError(t, err)
		assert.Equal(t, RecordKey("AAAAAAAD"), key)
		_, _, err = scan.Scan(context.Background())
		assert.Error(t, io.EOF, err)
	}
	{
		dir := path.Join(os.TempDir(), xlog.GenReqId())
		os.MkdirAll(dir, 0755)
		defer os.RemoveAll(dir)
		file := NewFile(bucket.NewLocalFS(dir), time.Hour, 3)
		file.Append(context.Background(), "AAAAAAAA", []byte("XXXXXXXXXXXXXXXX"))
		file.Append(context.Background(), "AAAAAAAB", []byte("XXXXXXXXXXXXXXXX"))
		file.Append(context.Background(), "AAAAAAAC", []byte("XXXXXXXXXXXXXXXX"))
		file.Append(context.Background(), "AAAAAAD", []byte("XXXXXXXXXXXXXXXX"))
		file.Flush(context.Background())

		scan := file.Scan(context.Background())
		_, _, err := scan.Scan(context.Background())
		assert.NoError(t, err)
		_, _, err = scan.Scan(context.Background())
		assert.NoError(t, err)
		_, _, err = scan.Scan(context.Background())
		assert.NoError(t, err)
		_, _, err = scan.Scan(context.Background())
		assert.NoError(t, err)
		_, _, err = scan.Scan(context.Background())
		assert.Error(t, io.EOF, err)
	}
}

func TestTime(t *testing.T) {

	{
		t1 := time.Date(2018, 4, 12, 17, 0, 0, 0, time.Local)
		t2 := time.Date(2018, 4, 12, 18, 0, 0, 0, time.Local)

		tt := _Time{
			Duration: time.Hour,
			Time:     t1,
			now:      func() time.Time { return t2 },
			offset:   time.Hour * 8,
		}

		assert.True(t, tt.Before(time.Hour))

		t2 = time.Date(2018, 4, 12, 17, 59, 0, 0, time.Local)
		assert.False(t, tt.Before(time.Hour))
		t2 = time.Date(2018, 4, 12, 19, 1, 0, 0, time.Local)
		assert.True(t, tt.Before(time.Hour*2))
		assert.False(t, tt.Before(time.Hour*3))

	}

}
