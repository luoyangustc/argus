package bucket

import (
	"context"
	"errors"
	"io"
	"testing"

	"github.com/stretchr/testify/assert"

	"qiniupkg.com/api.v7/kodo"
)

func TestBucketList(t *testing.T) {
	// b := Bucket{
	// 	BucketConfig: BucketConfig{
	// 		Config: kodo.Config{
	// 			AccessKey: "",
	// 			SecretKey: "",
	// 		},
	// 		Zone:   0,
	// 		Bucket: "argus-image-group",
	// 		// Bucket: "ava-model",
	// 		Prefix: "",
	// 	},
	// }

	// count, err := b.Count(context.Background())
	// t.Fatalf("COUNT: %d %v\n", count, err)
}

func TestEntryIter(t *testing.T) {

	{
		var entries []kodo.ListItem
		listFunc :=
			func(ctx context.Context, marker string, limit int) (
				[]kodo.ListItem, string, error) {
				return entries, "", io.EOF
			}
		iter := &EntryIter{listFunc: listFunc, limit: 4}
		_, _, ok := iter.Next(context.Background())
		assert.False(t, ok)
		assert.NoError(t, iter.Error())
	}
	{
		var entries []kodo.ListItem
		err := errors.New("ERR")
		listFunc :=
			func(ctx context.Context, marker string, limit int) (
				[]kodo.ListItem, string, error) {
				return entries, "", err
			}
		iter := &EntryIter{listFunc: listFunc, limit: 4}
		_, _, ok := iter.Next(context.Background())
		assert.False(t, ok)
		assert.Error(t, err, iter.Error())
	}
	{
		var entries []kodo.ListItem
		var err error
		listFunc :=
			func(ctx context.Context, marker string, limit int) (
				[]kodo.ListItem, string, error) {
				return entries, "", err
			}
		iter := &EntryIter{listFunc: listFunc, limit: 4}

		entries = []kodo.ListItem{{Key: "A"}, {Key: "B"}, {Key: "C"}, {Key: "D"}}
		item, _, ok := iter.Next(context.Background())
		assert.True(t, ok)
		assert.Equal(t, "A", item.Key)
		item, _, ok = iter.Next(context.Background())
		assert.True(t, ok)
		assert.Equal(t, "B", item.Key)
		item, _, ok = iter.Next(context.Background())
		assert.True(t, ok)
		assert.Equal(t, "C", item.Key)
		item, _, ok = iter.Next(context.Background())
		assert.True(t, ok)
		assert.Equal(t, "D", item.Key)

		entries = []kodo.ListItem{{Key: "E"}, {Key: "F"}}
		item, _, ok = iter.Next(context.Background())
		assert.True(t, ok)
		assert.Equal(t, "E", item.Key)
		item, _, ok = iter.Next(context.Background())
		assert.True(t, ok)
		assert.Equal(t, "F", item.Key)

		entries = nil
		err = io.EOF
		_, _, ok = iter.Next(context.Background())
		assert.False(t, ok)
		assert.NoError(t, iter.Error())
		_, _, ok = iter.Next(context.Background())
		assert.False(t, ok)

	}

}
