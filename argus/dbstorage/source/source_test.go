package source

import (
	"bytes"
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"qiniu.com/argus/dbstorage/proto"
)

func Test_Csv(t *testing.T) {
	ctx := context.Background()
	csv := NewCsvSource([]byte(""))
	line, err := csv.GetInfo(ctx)
	assert.Nil(t, err)
	assert.Equal(t, 0, line)

	csv = NewCsvSource([]byte("id1\nid2"))
	line, err = csv.GetInfo(ctx)
	assert.Equal(t, "csv file of the task must contains at least two columns for id and uri", err.Error())
	assert.Equal(t, 0, line)

	csv = NewCsvSource([]byte("id1,uri1\nid2"))
	line, err = csv.GetInfo(ctx)
	assert.NotNil(t, 0, err)
	assert.Equal(t, 0, line)

	csv = NewCsvSource([]byte("id1,uri1\nid2,uri2"))
	line, err = csv.GetInfo(ctx)
	assert.Nil(t, err)
	assert.Equal(t, 2, line)

	ch, err := csv.Read(ctx, func(i int) proto.ImageProcess { return proto.NOT_HANDLED })
	assert.Nil(t, err)
	task := <-ch
	assert.Equal(t, proto.ImageId("id1"), task.Id)
	assert.Equal(t, proto.ImageURI("uri1"), task.URI)
	task = <-ch
	assert.Equal(t, proto.ImageId("id2"), task.Id)
	assert.Equal(t, proto.ImageURI("uri2"), task.URI)
}

func Test_Json(t *testing.T) {
	ctx := context.Background()
	json := NewJsonSource([]byte(""))
	line, err := json.GetInfo(ctx)
	assert.Nil(t, err)
	assert.Equal(t, 0, line)

	json = NewJsonSource([]byte(`abcd`))
	line, err = json.GetInfo(ctx)
	assert.Equal(t, "line 1 : not correct json format", err.Error())
	assert.Equal(t, 0, line)

	json = NewJsonSource([]byte(`{"image":{"test":"abc"}}`))
	line, err = json.GetInfo(ctx)
	assert.Equal(t, "line 1 : must have node 'image', and have nodes 'id' and 'uri' inside", err.Error())
	assert.Equal(t, 0, line)

	var buf bytes.Buffer
	buf.WriteString(`{"image":{"id":"id1","uri":"uri1"}}`)
	buf.WriteString("\n\n")
	buf.WriteString(`{"image":{"id":"id2","uri":"uri2"}}`)
	json = NewJsonSource(buf.Bytes())
	line, err = json.GetInfo(ctx)
	assert.Nil(t, err)
	assert.Equal(t, 2, line)

	ch, err := json.Read(ctx, func(i int) proto.ImageProcess { return proto.NOT_HANDLED })
	assert.Nil(t, err)
	task := <-ch
	assert.Equal(t, proto.ImageId("id1"), task.Id)
	assert.Equal(t, proto.ImageURI("uri1"), task.URI)
	task = <-ch
	assert.Equal(t, proto.ImageId("id2"), task.Id)
	assert.Equal(t, proto.ImageURI("uri2"), task.URI)
}
