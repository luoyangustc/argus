package eval

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestCfgUnmarshal(t *testing.T) {
	var v EvalConfig
	json.Unmarshal([]byte(`{
    "batch_size": 1,
    "binaryproto_file": "http://oqgascup5.com0.z0.glb.qiniucdn.com/classify/binaryproto_file",
    "caffemodel_file": "http://oqgascup5.com0.z0.glb.qiniucdn.com/classify/caffemodel_file",
    "image_width": 224,
    "prototxt_file": "http://oqgascup5.com0.z0.glb.qiniucdn.com/classify/prototxt_file",
    "synset_file": "http://oqgascup5.com0.z0.glb.qiniucdn.com/classify/synset_file",
    "taglist_file": "http://oqgascup5.com0.z0.glb.qiniucdn.com/classify/taglist_file"
  }`), &v)
	assert.Equal(t, v.ImageWidth, 224)
}
