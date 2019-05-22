package video

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"io/ioutil"
	"os"
	"path"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	URI "qiniu.com/argus/argus/com/uri"
)

func TestFileSaver(t *testing.T) {

	var (
		saveSpace          = "/tmp/filesaver"
		saveAddress        = "http://localhost:8088"
		ctx                = context.Background()
		uid         uint32 = 1234
		uri                = URI.DataURIPrefix + base64.StdEncoding.EncodeToString([]byte("test"))
	)

	assert.Nil(t, os.Mkdir(saveSpace, 0777))
	defer os.RemoveAll(saveSpace)

	t.Run("fileSaver", func(t *testing.T) {
		cfg := FileSaveConfig{
			SaveSpace:   saveSpace,
			SaveAddress: saveAddress,
		}

		fileSaver := NewFileSaver(cfg)
		assert.NotNil(t, fileSaver)
		param := struct {
			Prefix string `json:"prefix"`
		}{
			Prefix: "",
		}
		p, _ := json.Marshal(param)
		hook, err := fileSaver.Get(ctx, uid, "vid", p)
		assert.Nil(t, err)
		assert.NotNil(t, hook)

		saver, err := hook.Get(ctx, "op_test")
		assert.Nil(t, err)
		assert.NotNil(t, saver)

		file, err := saver.Save(ctx, 100, uri)
		assert.Nil(t, err)
		assert.Equal(t, saveAddress+"/vid/100", file)
		content, err := ioutil.ReadFile(path.Join(saveSpace, strings.TrimPrefix(file, saveAddress+"/")))
		assert.Nil(t, err)
		assert.Equal(t, "test", string(content))
	})

	t.Run("带prefix的fileSaver", func(t *testing.T) {
		cfg := FileSaveConfig{
			SaveSpace:   saveSpace,
			SaveAddress: saveAddress,
		}

		fileSaver := NewFileSaver(cfg)
		assert.NotNil(t, fileSaver)
		param := struct {
			Prefix string `json:"prefix"`
		}{
			Prefix: "prefix",
		}
		p, _ := json.Marshal(param)
		hook, err := fileSaver.Get(ctx, uid, "vid", p)
		assert.Nil(t, err)
		assert.NotNil(t, hook)

		saver, err := hook.Get(ctx, "op_test")
		assert.Nil(t, err)
		assert.NotNil(t, saver)

		file, err := saver.Save(ctx, 100, uri)
		assert.Nil(t, err)
		assert.Equal(t, saveAddress+"/prefix/vid/100", file)
		content, err := ioutil.ReadFile(path.Join(saveSpace, strings.TrimPrefix(file, saveAddress+"/")))
		assert.Nil(t, err)
		assert.Equal(t, "test", string(content))
	})

	t.Run("带daily的fileSaver", func(t *testing.T) {
		cfg := FileSaveConfig{
			SaveSpace:   saveSpace,
			SaveAddress: saveAddress,
			DailyFolder: true,
		}

		fileSaver := NewFileSaver(cfg)
		assert.NotNil(t, fileSaver)
		param := struct {
			Prefix string `json:"prefix"`
		}{
			Prefix: "prefix",
		}
		p, _ := json.Marshal(param)
		hook, err := fileSaver.Get(ctx, uid, "vid", p)
		assert.Nil(t, err)
		assert.NotNil(t, hook)

		saver, err := hook.Get(ctx, "op_test")
		assert.Nil(t, err)
		assert.NotNil(t, saver)

		file, err := saver.Save(ctx, 100, uri)
		assert.Nil(t, err)
		assert.Equal(t, saveAddress+"/"+time.Now().Format("20060102")+"/prefix/vid/100", file)
		content, err := ioutil.ReadFile(path.Join(saveSpace, strings.TrimPrefix(file, saveAddress+"/")))
		assert.Nil(t, err)
		assert.Equal(t, "test", string(content))
	})

}
