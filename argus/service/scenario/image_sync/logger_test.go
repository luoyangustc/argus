package image_sync

// import (
// 	"github.com/stretchr/testify/assert"
// 	"io/ioutil"
// 	"os"
// 	"testing"
// )

// func TestLogger(t *testing.T) {
// 	var lg *ImgLogger
// 	{
// 		lg = NewLogger("method", "TestLogger")
// 		err := lg.Log("recode", "one line")
// 		assert.Nil(t, err)
// 	}
// 	var logdir = "./testlog"
// 	_ = os.Mkdir(logdir, 0775)
// 	defer os.RemoveAll(logdir)
// 	err := InitLogger(logConfig{
// 		LogDir: logdir,
// 	})
// 	assert.Nil(t, err)
// 	lg = NewLogger("test_log", "logger")
// 	_ = lg.Log("service_test", "image_sync")
// 	defaultLogWriter.Flush()
// 	files, _ := ioutil.ReadDir(logdir)
// 	assert.True(t, len(files) > 0)
// 	assert.True(t, files[0].Size() > int64(0))
// }
