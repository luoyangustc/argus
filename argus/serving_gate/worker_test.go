package gate

import (
	"context"
	"net/http"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"

	"qiniu.com/argus/atserving/model"
)

func TestWorker(t *testing.T) {
	w := NewWorker("")

	{
		resps, err := w.Do(
			context.Background(),
			time.Second*10,
			[]model.TaskReq{model.EvalRequest{}},
			func(ctx context.Context, id, cmd string, version *string, body []byte) error {
				go w.Handle(
					ctx,
					id,
					&model.ResponseMessage{
						StatusCode: http.StatusOK,
						Header: http.Header{
							model.KEY_DURATION: []string{time.Second.String()},
						},
					},
				)
				return nil
			},
		)
		assert.NoError(t, err, "do produce")
		assert.Equal(t, http.StatusOK, resps[0].StatusCode)
		duration, _ := time.ParseDuration(resps[0].Header.Get(model.KEY_DURATION))
		assert.Equal(t, time.Second, duration)
	}

	{
		err := w.Handle(context.Background(), "foo",
			&model.ResponseMessage{
				Header: http.Header{
					model.KEY_DURATION: []string{time.Second.String()},
				},
			},
		)
		assert.Equal(t, "overdue", err.Error())
	}
}
