package server

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestGetAppName(t *testing.T) {
	assert.Equal(t, "ava-facex-detect", getAppName("/v1/eval/facex-detect"))
	assert.Equal(t, "ava-facex-detect", getAppName("/v1/eval/facex-detect/v1"))
}

func TestFixBodyBuf(t *testing.T) {
	s := server{mock: true, cfg: Config{StsHost: "http://10.200.30.13:5555"}}
	t.Run("普通url", func(t *testing.T) {
		body := `{"data":{"uri":"http://baidu.com"}}`
		body2 := s.fixBodyBuf([]byte(body))
		assert.Equal(t, string(body), string(body2))
	})
	t.Run("sts url", func(t *testing.T) {
		body := `{"data":{"uri":"sts://10.200.30.15:5555/v1/file/CAAAAK_YnJGd10QV"}}`
		body2 := s.fixBodyBuf([]byte(body))
		assert.Equal(t, "{\"data\":{\"uri\":\"data:application/octet-stream;base64,Ym9keShodHRwOi8vMTAuMjAwLjMwLjE1OjU1NTUvdjEvZmlsZS9DQUFBQUtfWW5KR2QxMFFWKQ==\"}}", string(body2))
	})
	t.Run("url array", func(t *testing.T) {
		body := `{"data":[{"uri":"sts://10.200.30.15:5555/v1/file/CAAAAK_YnJGd10QV"}]}`
		body2 := s.fixBodyBuf([]byte(body))
		assert.Equal(t, "{\"data\":[{\"uri\":\"data:application/octet-stream;base64,Ym9keShodHRwOi8vMTAuMjAwLjMwLjE1OjU1NTUvdjEvZmlsZS9DQUFBQUtfWW5KR2QxMFFWKQ==\"}]}", string(body2))
	})
	t.Run("cs url", func(t *testing.T) {
		body := `{"data":{"uri":"http://10.200.30.13:10000/oygv408z3.bkt.clouddn.com/serving/terror-detect/set20180416/0000000000001041.jpg?e=1532590504\u0026token=0tf5awMVxwf8WrEvrjtbiZrdRZRJU-91JgCqTOC8:iq9zCdeYo1_j7FC2kFrGbXHFjGI"}}`
		body2 := s.fixBodyBuf([]byte(body))
		assert.Equal(t, "{\"data\":{\"uri\":\"data:application/octet-stream;base64,Ym9keShodHRwOi8vMTAuMjAwLjMwLjEzOjU1NTUvdjEvZmV0Y2g/dXJpPWh0dHAlM0ElMkYlMkYxMC4yMDAuMzAuMTMlM0ExMDAwMCUyRm95Z3Y0MDh6My5ia3QuY2xvdWRkbi5jb20lMkZzZXJ2aW5nJTJGdGVycm9yLWRldGVjdCUyRnNldDIwMTgwNDE2JTJGMDAwMDAwMDAwMDAwMTA0MS5qcGclM0ZlJTNEMTUzMjU5MDUwNCUyNnRva2VuJTNEMHRmNWF3TVZ4d2Y4V3JFdnJqdGJpWnJkUlpSSlUtOTFKZ0NxVE9DOCUzQWlxOXpDZGVZbzFfajdGQzJrRnJHYlhIRmpHSSk=\"}}", string(body2))
	})
}
